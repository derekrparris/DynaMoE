uniffi::setup_scaffolding!();

use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Debug, uniffi::Error)]
pub enum EngineError {
    FileError { details: String },
    MmapError { details: String },
    ParseError { details: String },
    TokenizerError { details: String },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::FileError { details } => write!(f, "File error: {}", details),
            EngineError::MmapError { details } => write!(f, "Mmap error: {}", details),
            EngineError::ParseError { details } => write!(f, "Parse error: {}", details),
            EngineError::TokenizerError { details } => write!(f, "Tokenizer error: {}", details),
        }
    }
}

// MARK: - Tokenizer Engine

#[derive(uniffi::Object)]
pub struct DynaMoeTokenizer {
    tokenizer: Tokenizer,
}

#[uniffi::export]
impl DynaMoeTokenizer {
    #[uniffi::constructor]
    pub fn new(tokenizer_path: String) -> Result<Arc<Self>, EngineError> {
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EngineError::TokenizerError { details: e.to_string() })?;
        Ok(Arc::new(Self { tokenizer }))
    }

    pub fn encode(&self, text: String) -> Result<Vec<u32>, EngineError> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| EngineError::TokenizerError { details: e.to_string() })?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: Vec<u32>) -> Result<String, EngineError> {
        let text = self.tokenizer.decode(&ids, true)
            .map_err(|e| EngineError::TokenizerError { details: e.to_string() })?;
        Ok(text)
    }
}

// MARK: - Model Engine Records & Objects

#[derive(uniffi::Record)]
pub struct ShardMetadata {
    pub index: u32,
    pub filename: String,
    pub base_address: u64,
    pub length: u64,
}

#[derive(uniffi::Record)]
pub struct TensorMetadata {
    pub name: String,
    pub shape_display: String,
    pub dtype: String,
    pub size_mb: f64,
    pub shard_index: u32,
    pub offset_start: u64,
    pub offset_end: u64,
    pub category: String,
    pub layer_index: Option<u32>,
    pub expert_id: Option<u32>,
}

#[derive(uniffi::Record)]
pub struct LayerSummary {
    pub layer_index: u32,
    pub total_tensors: u32,
    pub routed_expert_count: u32,
    pub total_size_mb: f64,
}

#[derive(uniffi::Record)]
pub struct ModelSummary {
    pub size_gb: f64,
    pub tensor_count: u32,
    pub layer_count: u32,
    pub max_expert_id: u32,
    pub shards: Vec<ShardMetadata>,
    pub tensors: Vec<TensorMetadata>,
    pub layers: Vec<LayerSummary>,
}

#[derive(Deserialize, Debug)]
struct WeightIndex {
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
    pub weight_map: BTreeMap<String, String>,
}

pub struct ShardHandle {
    pub filename: String,
    pub mmap: Mmap,
}

fn try_read_index_from_file(path: &Path) -> Option<(WeightIndex, PathBuf)> {
    if !path.is_file() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    let index: WeightIndex = serde_json::from_str(&content).ok()?;
    if index.weight_map.is_empty() {
        return None;
    }

    let parent = path.parent().unwrap_or(Path::new(""));

    // Case 1: Hugging Face cache 'blobs' directory (e.g. models--org--repo/blobs/<hash>)
    // The named shard symlinks live in models--org--repo/snapshots/<commit_id>/
    if parent.file_name().and_then(|s| s.to_str()) == Some("blobs") {
        if let Some(hub_dir) = parent.parent() {
            let snapshots_dir = hub_dir.join("snapshots");
            if snapshots_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snapshot_path = entry.path();
                        if snapshot_path.is_dir() {
                            if let Some(first_shard) = index.weight_map.values().next() {
                                if snapshot_path.join(first_shard).exists() {
                                    return Some((index, snapshot_path));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Case 2: Parent directly contains the shards
    if let Some(first_shard) = index.weight_map.values().next() {
        if parent.join(first_shard).exists() {
            return Some((index, parent.to_path_buf()));
        }
    }

    Some((index, parent.to_path_buf()))
}

fn resolve_model_index(target_path: &Path) -> Option<(WeightIndex, PathBuf)> {
    // 1. Direct file check
    if target_path.is_file() {
        if let Some(res) = try_read_index_from_file(target_path) {
            return Some(res);
        }
    }

    let search_dir = if target_path.is_dir() {
        target_path
    } else {
        target_path.parent().unwrap_or(Path::new(""))
    };

    // 2. Direct model.safetensors.index.json
    let direct_index = search_dir.join("model.safetensors.index.json");
    if let Some(res) = try_read_index_from_file(&direct_index) {
        return Some(res);
    }

    // 3. Hugging Face hub root containing snapshots/
    let snapshots_dir = search_dir.join("snapshots");
    if snapshots_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let snapshot_path = entry.path();
                if snapshot_path.is_dir() {
                    let snapshot_index = snapshot_path.join("model.safetensors.index.json");
                    if let Some(res) = try_read_index_from_file(&snapshot_index) {
                        return Some(res);
                    }
                }
            }
        }
    }

    // 4. Any *.json files in search_dir that parse as WeightIndex
    if let Ok(entries) = std::fs::read_dir(search_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(res) = try_read_index_from_file(&p) {
                    return Some(res);
                }
            }
        }
    }

    None
}

fn parse_layer_and_expert(name: &str) -> (String, Option<u32>, Option<u32>) {
    let mut layer_idx = None;
    let mut expert_idx = None;

    if let Some(pos) = name.find("layers.") {
        let rest = &name[pos + 7..];
        if let Some(end_pos) = rest.find('.') {
            if let Ok(idx) = rest[..end_pos].parse::<u32>() {
                layer_idx = Some(idx);
            }
        }
    }

    if let Some(pos) = name.find("experts.") {
        let rest = &name[pos + 8..];
        if let Some(end_pos) = rest.find('.') {
            if let Ok(idx) = rest[..end_pos].parse::<u32>() {
                expert_idx = Some(idx);
            }
        }
    }

    let category = if name.contains("embed_tokens") || name.contains("wte") {
        "Embedding".to_string()
    } else if name.contains("lm_head") {
        "LM Head".to_string()
    } else if name.contains("self_attn") || name.contains("attention") {
        "Self-Attention".to_string()
    } else if (name.contains("gate") || name.contains("router")) && name.contains("mlp") && expert_idx.is_none() {
        "MoE Router".to_string()
    } else if name.contains("shared_expert") {
        "Shared Expert".to_string()
    } else if expert_idx.is_some() || name.contains("experts") {
        if let Some(exp) = expert_idx {
            format!("Routed Expert #{}", exp)
        } else {
            "Routed Expert".to_string()
        }
    } else if name.contains("layernorm") || name.contains("norm") {
        "LayerNorm".to_string()
    } else if name.contains("mlp") {
        "Dense MLP".to_string()
    } else {
        "Other".to_string()
    };

    (category, layer_idx, expert_idx)
}

#[derive(uniffi::Object)]
pub struct DynaMoeEngine {
    shards: Vec<ShardHandle>,
}

#[uniffi::export]
impl DynaMoeEngine {
    #[uniffi::constructor]
    pub fn new(file_path: String) -> Result<Arc<Self>, EngineError> {
        let path = PathBuf::from(&file_path);
        let resolved_target = std::fs::canonicalize(&path).unwrap_or(path.clone());

        let mut shard_handles = Vec::new();

        // 1. Try to resolve multi-shard model via WeightIndex
        if let Some((index, base_dir)) = resolve_model_index(&path).or_else(|| resolve_model_index(&resolved_target)) {
            let mut unique_shards: Vec<String> = index.weight_map.values().cloned().collect();
            unique_shards.sort();
            unique_shards.dedup();

            for shard_name in unique_shards {
                let raw_shard_path = base_dir.join(&shard_name);
                let resolved_path = std::fs::canonicalize(&raw_shard_path).unwrap_or(raw_shard_path);

                let file = File::open(&resolved_path).map_err(|e| EngineError::FileError {
                    details: format!("Missing shard file '{}' at path {:?}: {}", shard_name, resolved_path, e),
                })?;

                let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| EngineError::MmapError {
                    details: format!("Failed to mmap shard '{}': {}", shard_name, e),
                })?;

                shard_handles.push(ShardHandle {
                    filename: shard_name,
                    mmap,
                });
            }
        } else if path.is_dir() {
            // 2. Directory without index: collect all *.safetensors files
            let mut safetensors_files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                        safetensors_files.push(p);
                    }
                }
            }
            safetensors_files.sort();

            if safetensors_files.is_empty() {
                return Err(EngineError::FileError {
                    details: format!("No .safetensors or index.json files found in directory {:?}", path),
                });
            }

            for p in safetensors_files {
                let filename = p.file_name().and_then(|s| s.to_str()).unwrap_or("shard.safetensors").to_string();
                let resolved_path = std::fs::canonicalize(&p).unwrap_or(p.clone());
                let file = File::open(&resolved_path).map_err(|e| EngineError::FileError {
                    details: format!("Failed to open file {:?}: {}", resolved_path, e),
                })?;
                let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| EngineError::MmapError {
                    details: format!("Failed to mmap file {:?}: {}", resolved_path, e),
                })?;
                shard_handles.push(ShardHandle { filename, mmap });
            }
        } else {
            // 3. Single file fallback
            let resolved_path = std::fs::canonicalize(&path).unwrap_or(path.clone());
            let file = File::open(&resolved_path).map_err(|e| EngineError::FileError { 
                details: format!("Failed to open file {:?}: {}", resolved_path, e) 
            })?;
            let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| EngineError::MmapError { details: e.to_string() })?;
            let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("model.safetensors").to_string();

            shard_handles.push(ShardHandle { filename, mmap });
        }

        if shard_handles.is_empty() {
            return Err(EngineError::FileError {
                details: format!("No valid model weights or shards could be loaded from {:?}", path),
            });
        }

        Ok(Arc::new(Self { shards: shard_handles }))
    }

    pub fn get_summary(&self) -> Result<ModelSummary, EngineError> {
        let mut total_bytes: u64 = 0;
        let mut shard_metadatas = Vec::new();
        let mut tensor_list = Vec::new();
        let mut layer_map: BTreeMap<u32, (u32, BTreeMap<u32, bool>, f64)> = BTreeMap::new();
        let mut max_expert_id = 0u32;

        for (shard_idx, shard) in self.shards.iter().enumerate() {
            let shard_len = shard.mmap.len() as u64;
            total_bytes += shard_len;

            shard_metadatas.push(ShardMetadata {
                index: shard_idx as u32,
                filename: shard.filename.clone(),
                base_address: shard.mmap.as_ptr() as u64,
                length: shard_len,
            });

            // Guard validation before parsing SafeTensors
            let shard_bytes = &shard.mmap[..];
            if shard_bytes.len() < 8 {
                return Err(EngineError::ParseError {
                    details: format!("Shard '{}' is too small ({} bytes) to be a valid SafeTensors file.", shard.filename, shard_bytes.len()),
                });
            }
            if shard_bytes.starts_with(b"version https://git-lfs") {
                return Err(EngineError::FileError {
                    details: format!("Shard '{}' is a Git LFS pointer text file, not actual model weights. Run 'git lfs pull' to fetch the real tensor binaries.", shard.filename),
                });
            }
            if shard_bytes.starts_with(b"{") || shard_bytes.starts_with(b"{\n") {
                return Err(EngineError::ParseError {
                    details: format!("Shard '{}' is a JSON text file, not a binary .safetensors weight file.", shard.filename),
                });
            }

            let tensors = SafeTensors::deserialize(&shard.mmap).map_err(|e| EngineError::ParseError {
                details: format!("Failed to parse shard '{}' ({:.2} MB): {:?}", shard.filename, shard_len as f64 / (1024.0 * 1024.0), e),
            })?;

            for name in tensors.names() {
                if let Ok(tensor) = tensors.tensor(name) {
                    let data_ptr = tensor.data().as_ptr() as usize;
                    let base_ptr = shard.mmap.as_ptr() as usize;
                    
                    let offset_start = (data_ptr - base_ptr) as u64;
                    let offset_end = offset_start + tensor.data().len() as u64;
                    let size_mb = tensor.data().len() as f64 / (1024.0 * 1024.0);

                    let (category, layer_index, expert_id) = parse_layer_and_expert(name);

                    if let Some(exp) = expert_id {
                        if exp + 1 > max_expert_id {
                            max_expert_id = exp + 1;
                        }
                    }

                    if let Some(l_idx) = layer_index {
                        let entry = layer_map.entry(l_idx).or_insert((0, BTreeMap::new(), 0.0));
                        entry.0 += 1;
                        if let Some(exp) = expert_id {
                            entry.1.insert(exp, true);
                        }
                        entry.2 += size_mb;
                    }

                    tensor_list.push(TensorMetadata {
                        name: name.to_string(),
                        shape_display: format!("{:?}", tensor.shape()),
                        dtype: format!("{:?}", tensor.dtype()),
                        size_mb,
                        shard_index: shard_idx as u32,
                        offset_start,
                        offset_end,
                        category,
                        layer_index,
                        expert_id,
                    });
                }
            }
        }

        tensor_list.sort_by(|a, b| a.name.cmp(&b.name));

        let layer_summaries: Vec<LayerSummary> = layer_map
            .into_iter()
            .map(|(layer_index, (total_tensors, experts, total_size_mb))| LayerSummary {
                layer_index,
                total_tensors,
                routed_expert_count: experts.len() as u32,
                total_size_mb,
            })
            .collect();

        let size_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        Ok(ModelSummary {
            size_gb,
            tensor_count: tensor_list.len() as u32,
            layer_count: layer_summaries.len() as u32,
            max_expert_id,
            shards: shard_metadatas,
            tensors: tensor_list,
            layers: layer_summaries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_index_from_blobs_and_snapshots() {
        let temp_dir = std::env::temp_dir().join(format!("dynamoe_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let hub_dir = temp_dir.join("models--test--moe");
        let blobs_dir = hub_dir.join("blobs");
        let snapshots_dir = hub_dir.join("snapshots").join("commit123");
        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::create_dir_all(&snapshots_dir).unwrap();

        // Create dummy shard in snapshots
        let shard_file = snapshots_dir.join("model-00001-of-00002.safetensors");
        std::fs::write(&shard_file, b"test dummy data").unwrap();

        // Create blob containing index JSON
        let index_json = r#"{
            "metadata": {"format": "pt"},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors"
            }
        }"#;
        let blob_hash_file = blobs_dir.join("99d9582580e39ac09f358abf2d146aa8fd1a6fc2");
        std::fs::write(&blob_hash_file, index_json).unwrap();

        // Test resolving index directly from blob path (reproducing the exact macOS symlink resolution scenario)
        let resolved = resolve_model_index(&blob_hash_file);
        assert!(resolved.is_some(), "Should resolve index from blobs path");
        let (index, base_dir) = resolved.unwrap();
        assert_eq!(index.weight_map.len(), 1);
        assert_eq!(base_dir, snapshots_dir);

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_real_qwen_path() {
        let blob_path = PathBuf::from("/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/blobs/99d9582580e39ac09f358abf2d146aa8fd1a6fc2");
        if blob_path.exists() {
            let res = resolve_model_index(&blob_path);
            println!("RESOLVED BLOB RESULT: {:?}", res.as_ref().map(|r| &r.1));
            assert!(res.is_some());
            let (_, base_dir) = res.unwrap();
            println!("BASE DIR: {:?}", base_dir);
            assert!(base_dir.to_string_lossy().contains("snapshots"));

            let engine = DynaMoeEngine::new(blob_path.to_string_lossy().to_string());
            assert!(engine.is_ok(), "Failed to create DynaMoeEngine: {:?}", engine.err());
            let engine = engine.unwrap();
            let summary = engine.get_summary();
            assert!(summary.is_ok(), "Failed to get summary: {:?}", summary.err());
            let summary = summary.unwrap();
            println!("SUCCESS! Loaded {} shards, {} tensors, {:.2} GB", summary.shards.len(), summary.tensor_count, summary.size_gb);
        }
    }
}