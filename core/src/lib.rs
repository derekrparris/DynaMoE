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
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| EngineError::TokenizerError { details: e.to_string() })?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: Vec<u32>) -> Result<String, EngineError> {
        let text = self.tokenizer.decode(&ids, false)
            .map_err(|e| EngineError::TokenizerError { details: e.to_string() })?;
        Ok(text)
    }
}

// MARK: - Model Engine Records & Objects

#[derive(uniffi::Record, Clone, Debug)]
pub struct ShardMetadata {
    pub index: u32,
    pub filename: String,
    pub base_address: u64,
    pub length: u64,
}

#[derive(uniffi::Record, Clone, Debug)]
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

#[derive(uniffi::Record, Clone, Debug)]
pub struct LayerSummary {
    pub layer_index: u32,
    pub total_tensors: u32,
    pub routed_expert_count: u32,
    pub total_size_mb: f64,
}

#[derive(uniffi::Record, Clone, Debug)]
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
    // If tensor belongs to Multi-Token Prediction (MTP) auxiliary head or visual encoder, do not treat as backbone layer
    if name.starts_with("mtp.") || name.starts_with("visual.") {
        let cat = if name.starts_with("mtp.") { "Multi-Token Prediction" } else { "Vision" };
        return (cat.to_string(), None, None);
    }

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
    } else if name.contains("shared_expert_gate") {
        "Shared Expert Gate".to_string()
    } else if (name.contains("mlp.gate.") || name.ends_with("mlp.gate")) && !name.contains("switch_mlp") && !name.contains("shared") && !name.contains("proj") {
        "MoE Router".to_string()
    } else if name.contains("shared_expert") {
        "Shared Expert".to_string()
    } else if expert_idx.is_some() || name.contains("experts") || name.contains("switch_mlp") {
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

#[derive(Deserialize, Debug)]
struct FlashMoEWeightsJson {
    #[serde(default)]
    pub tensors: BTreeMap<String, FlashMoETensorEntry>,
}

#[derive(Deserialize, Debug)]
struct FlashMoETensorEntry {
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<u64>,
    pub dtype: String,
}

#[derive(Deserialize, Debug)]
struct FlashMoELayoutJson {
    pub expert_size: u64,
    pub num_layers: u32,
    pub num_experts: u32,
    pub components: Vec<FlashMoEComponentEntry>,
}

#[derive(Deserialize, Debug)]
struct FlashMoEComponentEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

fn try_load_flash_moe(search_path: &Path) -> Option<(Vec<ShardHandle>, Vec<TensorMetadata>)> {
    let base_dir = if search_path.is_file() {
        search_path.parent().unwrap_or(Path::new(""))
    } else {
        search_path
    };

    let mut candidate_dirs = vec![base_dir.to_path_buf()];
    let snapshots_dir = base_dir.join("snapshots");
    if snapshots_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    candidate_dirs.push(entry.path());
                }
            }
        }
    }

    for dir in candidate_dirs {
        let weights_json_path = dir.join("model_weights.json");
        let weights_bin_path = dir.join("model_weights.bin");
        let packed_experts_dir = dir.join("packed_experts");
        let layout_json_path = packed_experts_dir.join("layout.json");

        if weights_json_path.is_file() && weights_bin_path.is_file() {
            let weights_json_data = std::fs::read_to_string(&weights_json_path).ok()?;
            let weights_json: FlashMoEWeightsJson = serde_json::from_str(&weights_json_data).ok()?;

            let mut shard_handles = Vec::new();

            // Shard 0: model_weights.bin
            let file0 = File::open(&weights_bin_path).ok()?;
            let mmap0 = unsafe { MmapOptions::new().map(&file0) }.ok()?;
            shard_handles.push(ShardHandle {
                filename: "model_weights.bin".to_string(),
                mmap: mmap0,
            });

            let mut tensor_list = Vec::new();

            // 1. Non-expert tensors from model_weights.json (Shard 0)
            for (name, entry) in weights_json.tensors {
                let (category, layer_index, expert_id) = parse_layer_and_expert(&name);
                let size_mb = entry.size as f64 / (1024.0 * 1024.0);
                tensor_list.push(TensorMetadata {
                    name,
                    shape_display: format!("{:?}", entry.shape),
                    dtype: entry.dtype,
                    size_mb,
                    shard_index: 0,
                    offset_start: entry.offset,
                    offset_end: entry.offset + entry.size,
                    category,
                    layer_index,
                    expert_id,
                });
            }

            // 2. Packed experts from packed_experts/
            if layout_json_path.is_file() {
                if let Ok(layout_data) = std::fs::read_to_string(&layout_json_path) {
                    if let Ok(layout) = serde_json::from_str::<FlashMoELayoutJson>(&layout_data) {
                        for l in 0..layout.num_layers {
                            let layer_bin_name = format!("layer_{:02}.bin", l);
                            let layer_bin_path = packed_experts_dir.join(&layer_bin_name);
                            if let Ok(file_l) = File::open(&layer_bin_path) {
                                if let Ok(mmap_l) = unsafe { MmapOptions::new().map(&file_l) } {
                                    let shard_idx = shard_handles.len() as u32;
                                    shard_handles.push(ShardHandle {
                                        filename: format!("packed_experts/{}", layer_bin_name),
                                        mmap: mmap_l,
                                    });

                                    for e in 0..layout.num_experts {
                                        let expert_base = (e as u64) * layout.expert_size;
                                        for comp in &layout.components {
                                            let tensor_name = format!("model.layers.{}.mlp.experts.{}.{}", l, e, comp.name);
                                            let offset_start = expert_base + comp.offset;
                                            let offset_end = offset_start + comp.size;
                                            let size_mb = comp.size as f64 / (1024.0 * 1024.0);

                                            tensor_list.push(TensorMetadata {
                                                name: tensor_name,
                                                shape_display: format!("{:?}", comp.shape),
                                                dtype: comp.dtype.clone(),
                                                size_mb,
                                                shard_index: shard_idx,
                                                offset_start,
                                                offset_end,
                                                category: format!("Routed Expert #{}", e),
                                                layer_index: Some(l),
                                                expert_id: Some(e),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if tensor_list.len() <= 2000 {
                tensor_list.sort_by(|a, b| a.name.cmp(&b.name));
            }
            return Some((shard_handles, tensor_list));
        }
    }

    None
}

#[derive(uniffi::Object)]
pub struct DynaMoeEngine {
    shards: Vec<ShardHandle>,
    custom_tensors: Option<Vec<TensorMetadata>>,
}

#[uniffi::export]
impl DynaMoeEngine {
    #[uniffi::constructor]
    pub fn new(file_path: String) -> Result<Arc<Self>, EngineError> {
        let path = PathBuf::from(&file_path);
        let resolved_target = std::fs::canonicalize(&path).unwrap_or(path.clone());

        // 0. Try FlashMoE Q4 Model Format
        if let Some((shards, tensors)) = try_load_flash_moe(&path).or_else(|| try_load_flash_moe(&resolved_target)) {
            return Ok(Arc::new(Self {
                shards,
                custom_tensors: Some(tensors),
            }));
        }

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

        Ok(Arc::new(Self { shards: shard_handles, custom_tensors: None }))
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
        }

        if let Some(ref custom) = self.custom_tensors {
            tensor_list = custom.clone();
            for t in &tensor_list {
                if let Some(exp) = t.expert_id {
                    if exp + 1 > max_expert_id {
                        max_expert_id = exp + 1;
                    }
                }
                if let Some(l_idx) = t.layer_index {
                    let entry = layer_map.entry(l_idx).or_insert((0, BTreeMap::new(), 0.0));
                    entry.0 += 1;
                    if let Some(exp) = t.expert_id {
                        entry.1.insert(exp, true);
                    }
                    entry.2 += t.size_mb;
                }
            }
        } else {
            for (shard_idx, shard) in self.shards.iter().enumerate() {
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
                    details: format!("Failed to parse shard '{}' ({:.2} MB): {:?}", shard.filename, shard_bytes.len() as f64 / (1024.0 * 1024.0), e),
                })?;

                for name in tensors.names() {
                    if let Ok(tensor) = tensors.tensor(name) {
                        let data_ptr = tensor.data().as_ptr() as usize;
                        let base_ptr = shard.mmap.as_ptr() as usize;
                        
                        let offset_start = (data_ptr - base_ptr) as u64;
                        let offset_end = offset_start + tensor.data().len() as u64;
                        let size_mb = tensor.data().len() as f64 / (1024.0 * 1024.0);

                        let (category, layer_index, expert_id) = parse_layer_and_expert(name);

                        let shape = tensor.shape();
                        // Stacked 3D expert tensor: e.g. switch_mlp with shape [num_experts, dim0, dim1]
                        if shape.len() == 3 && shape[0] > 1 && (name.contains("switch_mlp") || name.contains("experts")) && expert_id.is_none() {
                            let num_experts = shape[0];
                            let per_expert_bytes = (tensor.data().len() / num_experts) as u64;
                            let per_expert_size_mb = size_mb / (num_experts as f64);
                            let sub_shape_display = format!("{:?}", &shape[1..]);

                            if let Some(l_idx) = layer_index {
                                let entry = layer_map.entry(l_idx).or_insert((0, BTreeMap::new(), 0.0));
                                for exp_id in 0..num_experts {
                                    entry.1.insert(exp_id as u32, true);
                                }
                                entry.0 += num_experts as u32;
                                entry.2 += size_mb;
                            }

                            if max_expert_id < num_experts as u32 {
                                max_expert_id = num_experts as u32;
                            }

                            for exp_id in 0..num_experts {
                                let exp_name = if name.contains("switch_mlp") {
                                    name.replace("switch_mlp", &format!("experts.{}", exp_id))
                                } else {
                                    name.replace("experts.", &format!("experts.{}.", exp_id))
                                };
                                let exp_offset_start = offset_start + (exp_id as u64) * per_expert_bytes;
                                let exp_offset_end = exp_offset_start + per_expert_bytes;

                                tensor_list.push(TensorMetadata {
                                    name: exp_name,
                                    shape_display: sub_shape_display.clone(),
                                    dtype: format!("{:?}", tensor.dtype()),
                                    size_mb: per_expert_size_mb,
                                    shard_index: shard_idx as u32,
                                    offset_start: exp_offset_start,
                                    offset_end: exp_offset_end,
                                    category: format!("Routed Expert #{}", exp_id),
                                    layer_index,
                                    expert_id: Some(exp_id as u32),
                                });
                            }
                        } else {
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
            }
        }

        if tensor_list.len() <= 2000 {
            tensor_list.sort_by(|a, b| a.name.cmp(&b.name));
        }

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
            let router_tensors: Vec<_> = summary.tensors.iter().filter(|t| t.name.ends_with("gate.weight") || t.name.contains("router") || t.category == "MoE Router").take(10).collect();
            for t in router_tensors {
                println!("ROUTER TENSOR: name={}, shape={}, dtype={}, shard={}, offset={}", t.name, t.shape_display, t.dtype, t.shard_index, t.offset_start);
            }
        }
    }

    #[test]
    fn test_real_ornith_path() {
        let snapshot_dir = PathBuf::from("/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c");
        let index_file = snapshot_dir.join("model.safetensors.index.json");
        if index_file.exists() {
            let engine = DynaMoeEngine::new(index_file.to_string_lossy().to_string());
            assert!(engine.is_ok(), "Failed to create DynaMoeEngine for Ornith: {:?}", engine.err());
            let engine = engine.unwrap();
            let summary = engine.get_summary();
            assert!(summary.is_ok(), "Failed to get summary for Ornith: {:?}", summary.err());
            let summary = summary.unwrap();
            println!("ORNITH SUCCESS! Loaded {} shards, {} tensors, {:.2} GB, {} layers, max expert ID {}", 
                     summary.shards.len(), summary.tensor_count, summary.size_gb, summary.layer_count, summary.max_expert_id);
            println!("=== LAYER 0 NON-EXPERT TENSORS ===");
            for t in summary.tensors.iter().filter(|t| t.layer_index == Some(0) && !t.name.contains("experts.")) {
                println!("  L0 (non-expert): name={}, shape={}, dtype={}, shard={}, offset={}", t.name, t.shape_display, t.dtype, t.shard_index, t.offset_start);
            }
            println!("=== LAYER 3 TENSORS ===");
            for t in summary.tensors.iter().filter(|t| t.layer_index == Some(3) && !t.name.contains("experts.")) {
                println!("  L3 (non-expert): name={}, shape={}, dtype={}, shard={}, offset={}", t.name, t.shape_display, t.dtype, t.shard_index, t.offset_start);
            }
            assert_eq!(summary.shards.len(), 16);
        }
    }

    #[test]
    fn test_real_flashmoe_q4_path() {
        let snapshot_dir = PathBuf::from("/Users/derekparris/.cache/huggingface/hub/models--alexintosh--Qwen3.5-35B-A3B-Q4-FlashMoE/snapshots/954605e54d19dca04d607421114b301ae9c2e061");
        if snapshot_dir.exists() {
            let engine = DynaMoeEngine::new(snapshot_dir.to_string_lossy().to_string());
            assert!(engine.is_ok(), "Failed to create DynaMoeEngine for FlashMoE: {:?}", engine.err());
            let engine = engine.unwrap();
            let summary = engine.get_summary();
            assert!(summary.is_ok(), "Failed to get summary for FlashMoE: {:?}", summary.err());
            let summary = summary.unwrap();
            println!("FLASHMOE SUCCESS! Loaded {} shards, {} tensors, {:.2} GB, {} layers, max expert ID {}", 
                     summary.shards.len(), summary.tensor_count, summary.size_gb, summary.layer_count, summary.max_expert_id);
            assert_eq!(summary.shards.len(), 41); // Shard 0 (model_weights.bin) + Shards 1..40 (layer_00.bin..layer_39.bin)
            assert_eq!(summary.layer_count, 40);
            assert_eq!(summary.max_expert_id, 256);
        }
    }


    #[test]
    fn test_ornith_tokenizer() {
        let tok_path = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c/tokenizer.json";
        if std::path::Path::new(tok_path).exists() {
            let tok = DynaMoeTokenizer::new(tok_path.to_string()).unwrap();
            let prompt = "<|im_start|>user\nHello, what is 2+2?<|im_end|>\n<|im_start|>assistant\n";
            let ids = tok.encode(prompt.to_string()).unwrap();
            println!("ENCODED IDS ({} tokens): {:?}", ids.len(), ids);
            for id in &ids {
                let piece = tok.decode(vec![*id]).unwrap_or_default();
                println!("  Token ID {}: {:?}", id, piece);
            }

            let garbled = "ブログ村|RFСТА,eg_INETdisplayTextalisesanitizeereumブログ村";
            let g_ids = tok.encode(garbled.to_string()).unwrap();
            println!("GARBLED 1 IDS: {:?}", g_ids);

            let garbled2 = "ereum{lng是何含义izedName----</icts";
            let g2_ids = tok.encode(garbled2.to_string()).unwrap();
            println!("GARBLED 2 IDS: {:?}", g2_ids);
        }
    }

    #[inline]
    fn bf16_to_f32(u: u16) -> f32 {
        f32::from_bits((u as u32) << 16)
    }

    #[inline]
    fn unpack_e4m3_rs(u: u8) -> f32 {
        let sign = (u >> 7) & 0x01;
        let exp = (u >> 3) & 0x0F;
        let mant = u & 0x07;
        let val = if exp == 0 {
            (mant as f32 / 8.0) * 0.015625
        } else {
            (1.0 + (mant as f32 / 8.0)) * 2.0f32.powi(exp as i32 - 7)
        };
        if sign != 0 { -val } else { val }
    }

    fn rmsnorm_rs(in_vec: &[f32], weight_mmap: &[u8], weight_offset: usize, out_vec: &mut [f32], eps: f32) {
        let dim = in_vec.len();
        let w_u16 = unsafe {
            std::slice::from_raw_parts(weight_mmap.as_ptr().add(weight_offset) as *const u16, dim)
        };
        let mut sum_sq = 0.0f32;
        for i in 0..dim {
            sum_sq += in_vec[i] * in_vec[i];
        }
        let inv_rms = 1.0 / ((sum_sq / dim as f32) + eps).sqrt();
        for i in 0..dim {
            let w = bf16_to_f32(w_u16[i]);
            out_vec[i] = in_vec[i] * inv_rms * w;
        }
    }

    fn gemv_bf16_rs(shard_mmap: &[u8], offset: usize, in_vec: &[f32], out_vec: &mut [f32], in_dim: usize, out_dim: usize) {
        let total_words = in_dim * out_dim;
        let w_u16 = unsafe {
            std::slice::from_raw_parts(shard_mmap.as_ptr().add(offset) as *const u16, total_words)
        };
        for row in 0..out_dim {
            let row_start = row * in_dim;
            let mut dot = 0.0f32;
            for col in 0..in_dim {
                dot += bf16_to_f32(w_u16[row_start + col]) * in_vec[col];
            }
            out_vec[row] = dot;
        }
    }

    fn gemv_fp8_rs(w_mmap: &[u8], w_offset: usize, s_mmap: &[u8], s_offset: usize, in_vec: &[f32], out_vec: &mut [f32], in_dim: usize, out_dim: usize) {
        let total_bytes = in_dim * out_dim;
        let w_bytes = &w_mmap[w_offset..w_offset + total_bytes];
        let s_u16 = unsafe {
            std::slice::from_raw_parts(s_mmap.as_ptr().add(s_offset) as *const u16, out_dim)
        };
        for row in 0..out_dim {
            let row_start = row * in_dim;
            let scale = bf16_to_f32(s_u16[row]);
            let mut dot = 0.0f32;
            for col in 0..in_dim {
                let raw_byte = w_bytes[row_start + col];
                dot += unpack_e4m3_rs(raw_byte) * in_vec[col];
            }
            out_vec[row] = dot * scale;
        }
    }

    #[test]
    fn test_cpu_layer0_forward() {
        let snapshot_dir = PathBuf::from("/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c");
        let index_file = snapshot_dir.join("model.safetensors.index.json");
        if !index_file.exists() { return; }

        let engine = DynaMoeEngine::new(index_file.to_string_lossy().to_string()).unwrap();
        let summary = engine.get_summary().unwrap();

        // 1. Embedding lookup for token 248045 (<|im_start|>)
        let embed_tensor = summary.tensors.iter().find(|t| t.name == "model.language_model.embed_tokens.weight").unwrap();
        let embed_mmap = &engine.shards[embed_tensor.shard_index as usize].mmap;
        let target_token_id: usize = 248045;
        let hidden_dim: usize = 2048;

        let mut h_0 = vec![0.0f32; hidden_dim];
        let embed_u16 = unsafe {
            std::slice::from_raw_parts(embed_mmap.as_ptr().add(embed_tensor.offset_start as usize) as *const u16, 248320 * hidden_dim)
        };
        for d in 0..hidden_dim {
            h_0[d] = bf16_to_f32(embed_u16[target_token_id * hidden_dim + d]);
        }
        println!("h_0 first 8 dims: {:?}", &h_0[0..8]);

        // 2. Layer 0 Input Layernorm
        let norm1 = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.input_layernorm.weight").unwrap();
        let norm1_mmap = &engine.shards[norm1.shard_index as usize].mmap;
        let mut x_norm1 = vec![0.0f32; hidden_dim];
        rmsnorm_rs(&h_0, norm1_mmap, norm1.offset_start as usize, &mut x_norm1, 1e-6);
        println!("x_norm1 first 8 dims: {:?}", &x_norm1[0..8]);

        // 3. Layer 0 in_proj_qkv (8192 x 2048)
        let in_qkv = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight").unwrap();
        let in_qkv_mmap = &engine.shards[in_qkv.shard_index as usize].mmap;
        let mut qkv = vec![0.0f32; 8192];
        gemv_bf16_rs(in_qkv_mmap, in_qkv.offset_start as usize, &x_norm1, &mut qkv, hidden_dim, 8192);
        println!("qkv first 8 dims (Q): {:?}", &qkv[0..8]);
        println!("qkv middle 8 dims (K): {:?}", &qkv[2048..2056]);
        println!("qkv value 8 dims (V): {:?}", &qkv[4096..4104]);

        // 4. conv1d
        let conv_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.conv1d.weight").unwrap();
        let conv_mmap = &engine.shards[conv_t.shard_index as usize].mmap;
        let conv_u16 = unsafe {
            std::slice::from_raw_parts(conv_mmap.as_ptr().add(conv_t.offset_start as usize) as *const u16, 8192 * 4)
        };
        for c in 0..4 {
            let w0 = bf16_to_f32(conv_u16[c * 4 + 0]);
            let w1 = bf16_to_f32(conv_u16[c * 4 + 1]);
            let w2 = bf16_to_f32(conv_u16[c * 4 + 2]);
            let w3 = bf16_to_f32(conv_u16[c * 4 + 3]);
            println!("conv channel {}: [w0={}, w1={}, w2={}, w3={}]", c, w0, w1, w2, w3);
        }
        let mut conv_out = vec![0.0f32; 8192];
        for c in 0..8192 {
            let w3 = bf16_to_f32(conv_u16[c * 4 + 3]);
            let s3 = qkv[c];
            let conv_val = w3 * s3; // first token, previous state is 0
            let silu_val = conv_val / (1.0 + (-conv_val).exp());
            conv_out[c] = silu_val;
        }
        println!("conv_out first 8 dims: {:?}", &conv_out[0..8]);

        // 5. in_proj_z (4096 x 2048)
        let in_z = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.in_proj_z.weight").unwrap();
        let in_z_mmap = &engine.shards[in_z.shard_index as usize].mmap;
        let mut z = vec![0.0f32; 4096];
        gemv_bf16_rs(in_z_mmap, in_z.offset_start as usize, &x_norm1, &mut z, hidden_dim, 4096);
        println!("z first 8 dims: {:?}", &z[0..8]);

        // 6. in_proj_a (32 x 2048) & in_proj_b (32 x 2048)
        let in_a = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.in_proj_a.weight").unwrap();
        let in_a_mmap = &engine.shards[in_a.shard_index as usize].mmap;
        let mut a_vec = vec![0.0f32; 32];
        gemv_bf16_rs(in_a_mmap, in_a.offset_start as usize, &x_norm1, &mut a_vec, hidden_dim, 32);

        let in_b = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.in_proj_b.weight").unwrap();
        let in_b_mmap = &engine.shards[in_b.shard_index as usize].mmap;
        let mut b_vec = vec![0.0f32; 32];
        gemv_bf16_rs(in_b_mmap, in_b.offset_start as usize, &x_norm1, &mut b_vec, hidden_dim, 32);

        println!("a_vec: {:?}", a_vec);
        println!("b_vec: {:?}", b_vec);

        // 7. L2-norm on Q (16 heads x 128) and K (16 heads x 128)
        let num_heads: usize = 16;
        let head_dim: usize = 128;
        for h in 0..num_heads {
            // Q head
            let q_base = h * head_dim;
            let mut q_sum_sq = 0.0f32;
            for i in 0..head_dim {
                let v = conv_out[q_base + i];
                q_sum_sq += v * v;
            }
            let inv_q = 1.0 / (q_sum_sq + 1e-6).sqrt();
            for i in 0..head_dim {
                conv_out[q_base + i] *= inv_q;
            }

            // K head
            let k_base = 2048 + (h * head_dim);
            let mut k_sum_sq = 0.0f32;
            for i in 0..head_dim {
                let v = conv_out[k_base + i];
                k_sum_sq += v * v;
            }
            let inv_k = 1.0 / (k_sum_sq + 1e-6).sqrt();
            for i in 0..head_dim {
                conv_out[k_base + i] *= inv_k;
            }
        }

        // 8. Linear Recurrence Step
        let a_log_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.A_log").unwrap();
        let a_log_mmap = &engine.shards[a_log_t.shard_index as usize].mmap;
        let a_log_u16 = unsafe {
            std::slice::from_raw_parts(a_log_mmap.as_ptr().add(a_log_t.offset_start as usize) as *const u16, 32)
        };

        let dt_bias_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.dt_bias").unwrap();
        let dt_bias_mmap = &engine.shards[dt_bias_t.shard_index as usize].mmap;
        let dt_bias_u16 = unsafe {
            std::slice::from_raw_parts(dt_bias_mmap.as_ptr().add(dt_bias_t.offset_start as usize) as *const u16, 32)
        };

        let lin_norm_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.norm.weight").unwrap();
        let lin_norm_mmap = &engine.shards[lin_norm_t.shard_index as usize].mmap;
        let lin_norm_u16 = unsafe {
            std::slice::from_raw_parts(lin_norm_mmap.as_ptr().add(lin_norm_t.offset_start as usize) as *const u16, 128)
        };

        let mut attn_ctx = vec![0.0f32; 4096];
        let mut state_matrices = vec![0.0f32; 32 * 128 * 128]; // [32, 128, 128]

        for h in 0..32 {
            let key_head_idx = h / 2;
            let q_base = key_head_idx * head_dim;
            let k_base = 2048 + (key_head_idx * head_dim);
            let v_base = 4096 + (h * head_dim);
            let z_base = h * head_dim;
            let out_base = h * head_dim;
            let state_base = h * head_dim * head_dim;

            let a_log_val = bf16_to_f32(a_log_u16[h]);
            let dt_bias_val = bf16_to_f32(dt_bias_u16[h]);
            let a_val = a_vec[h];
            let b_val = b_vec[h];

            let x = a_val + dt_bias_val;
            let dt = if x > 20.0 { x } else if x < -20.0 { x.exp() } else { (1.0 + x.exp()).ln() };
            let alpha = (-a_log_val.exp() * dt).exp();
            let beta = 1.0 / (1.0 + (-b_val).exp());

            // Sk = S_prev * k
            let mut sk = [0.0f32; 128];
            for i in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..head_dim {
                    sum += state_matrices[state_base + (i * head_dim) + j] * conv_out[k_base + j];
                }
                sk[i] = sum;
            }

            let mut y = [0.0f32; 128];
            let mut sum_sq = 0.0f32;
            for i in 0..head_dim {
                let vi = conv_out[v_base + i];
                let ui = vi - (alpha * sk[i]);
                let beta_ui = beta * ui;

                let mut yi = 0.0f32;
                for j in 0..head_dim {
                    let s_idx = state_base + (i * head_dim) + j;
                    let s_old = state_matrices[s_idx];
                    let kj = conv_out[k_base + j];
                    let s_new = (alpha * s_old) + (beta_ui * kj);
                    state_matrices[s_idx] = s_new;
                    yi += s_new * conv_out[q_base + j];
                }
                y[i] = yi;
                sum_sq += yi * yi;
            }

            let inv_rms = 1.0 / ((sum_sq / head_dim as f32) + 1e-6).sqrt();
            for i in 0..head_dim {
                let norm_w = bf16_to_f32(lin_norm_u16[i]);
                let y_norm = y[i] * inv_rms * norm_w;
                let z_val = z[z_base + i];
                let silu_z = z_val / (1.0 + (-z_val).exp());
                attn_ctx[out_base + i] = y_norm * silu_z;
            }
        }
        println!("attn_ctx first 8 dims: {:?}", &attn_ctx[0..8]);

        // 9. out_proj (2048 x 4096)
        let out_proj_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.linear_attn.out_proj.weight").unwrap();
        let out_proj_mmap = &engine.shards[out_proj_t.shard_index as usize].mmap;
        let mut attn_out = vec![0.0f32; hidden_dim];
        gemv_bf16_rs(out_proj_mmap, out_proj_t.offset_start as usize, &attn_ctx, &mut attn_out, 4096, hidden_dim);
        println!("attn_out first 8 dims: {:?}", &attn_out[0..8]);

        // 10. Residual 1 (h_mid = h_0 + attn_out)
        let mut h_mid = vec![0.0f32; hidden_dim];
        for d in 0..hidden_dim {
            h_mid[d] = h_0[d] + attn_out[d];
        }
        println!("h_mid first 8 dims: {:?}", &h_mid[0..8]);

        // 11. post_attention_layernorm
        let norm2 = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.post_attention_layernorm.weight").unwrap();
        let norm2_mmap = &engine.shards[norm2.shard_index as usize].mmap;
        let mut x_norm2 = vec![0.0f32; hidden_dim];
        rmsnorm_rs(&h_mid, norm2_mmap, norm2.offset_start as usize, &mut x_norm2, 1e-6);
        println!("x_norm2 first 8 dims: {:?}", &x_norm2[0..8]);

        // 12. Router (256 x 2048)
        let router_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.gate.weight").unwrap();
        let router_mmap = &engine.shards[router_t.shard_index as usize].mmap;
        let mut router_logits = vec![0.0f32; 256];
        gemv_bf16_rs(router_mmap, router_t.offset_start as usize, &x_norm2, &mut router_logits, hidden_dim, 256);

        // Top 8
        let mut router_pairs: Vec<(usize, f32)> = router_logits.iter().cloned().enumerate().collect();
        router_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top8: Vec<(usize, f32)> = router_pairs.into_iter().take(8).collect();
        println!("Top 8 router experts: {:?}", top8);

        // Softmax on top 8
        let max_l = top8[0].1;
        let exp_sum: f32 = top8.iter().map(|(_, l)| (l - max_l).exp()).sum();
        let top8_weights: Vec<(usize, f32)> = top8.iter().map(|(id, l)| (*id, (l - max_l).exp() / exp_sum)).collect();
        println!("Top 8 normalized weights: {:?}", top8_weights);

        // 13. Shared expert gate
        let shared_gate_t = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert_gate.weight").unwrap();
        let shared_gate_mmap = &engine.shards[shared_gate_t.shard_index as usize].mmap;
        let shared_gate_u16 = unsafe {
            std::slice::from_raw_parts(shared_gate_mmap.as_ptr().add(shared_gate_t.offset_start as usize) as *const u16, hidden_dim)
        };
        let mut sg_dot = 0.0f32;
        for d in 0..hidden_dim {
            sg_dot += bf16_to_f32(shared_gate_u16[d]) * x_norm2[d];
        }
        let shared_w = 1.0 / (1.0 + (-sg_dot).exp());
        println!("Shared expert weight: {}", shared_w);

        // 14. MLP Accumulation
        let mut h_mlp = vec![0.0f32; hidden_dim];
        let inter_dim: usize = 512;

        // Shared expert
        let sg_w = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight").unwrap();
        let sg_s = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_scale").unwrap();
        let su_w = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert.up_proj.weight").unwrap();
        let su_s = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert.up_proj.weight_scale").unwrap();
        let sd_w = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert.down_proj.weight").unwrap();
        let sd_s = summary.tensors.iter().find(|t| t.name == "model.language_model.layers.0.mlp.shared_expert.down_proj.weight_scale").unwrap();

        let mut gate_vec = vec![0.0f32; inter_dim];
        let mut up_vec = vec![0.0f32; inter_dim];
        let mut inter_vec = vec![0.0f32; inter_dim];
        let mut down_vec = vec![0.0f32; hidden_dim];

        gemv_fp8_rs(&engine.shards[sg_w.shard_index as usize].mmap, sg_w.offset_start as usize,
                    &engine.shards[sg_s.shard_index as usize].mmap, sg_s.offset_start as usize,
                    &x_norm2, &mut gate_vec, hidden_dim, inter_dim);
        gemv_fp8_rs(&engine.shards[su_w.shard_index as usize].mmap, su_w.offset_start as usize,
                    &engine.shards[su_s.shard_index as usize].mmap, su_s.offset_start as usize,
                    &x_norm2, &mut up_vec, hidden_dim, inter_dim);

        for i in 0..inter_dim {
            let g = gate_vec[i];
            let silu_g = g / (1.0 + (-g).exp());
            inter_vec[i] = silu_g * up_vec[i];
        }

        gemv_fp8_rs(&engine.shards[sd_w.shard_index as usize].mmap, sd_w.offset_start as usize,
                    &engine.shards[sd_s.shard_index as usize].mmap, sd_s.offset_start as usize,
                    &inter_vec, &mut down_vec, inter_dim, hidden_dim);

        for d in 0..hidden_dim {
            h_mlp[d] += down_vec[d] * shared_w;
        }

        // Top 8 routed experts
        for (exp_id, p_k) in top8_weights {
            let eg_w_name = format!("model.language_model.layers.0.mlp.experts.{}.gate_proj.weight", exp_id);
            let eg_s_name = format!("model.language_model.layers.0.mlp.experts.{}.gate_proj.weight_scale", exp_id);
            let eu_w_name = format!("model.language_model.layers.0.mlp.experts.{}.up_proj.weight", exp_id);
            let eu_s_name = format!("model.language_model.layers.0.mlp.experts.{}.up_proj.weight_scale", exp_id);
            let ed_w_name = format!("model.language_model.layers.0.mlp.experts.{}.down_proj.weight", exp_id);
            let ed_s_name = format!("model.language_model.layers.0.mlp.experts.{}.down_proj.weight_scale", exp_id);

            let eg_w = summary.tensors.iter().find(|t| t.name == eg_w_name).unwrap();
            let eg_s = summary.tensors.iter().find(|t| t.name == eg_s_name).unwrap();
            let eu_w = summary.tensors.iter().find(|t| t.name == eu_w_name).unwrap();
            let eu_s = summary.tensors.iter().find(|t| t.name == eu_s_name).unwrap();
            let ed_w = summary.tensors.iter().find(|t| t.name == ed_w_name).unwrap();
            let ed_s = summary.tensors.iter().find(|t| t.name == ed_s_name).unwrap();

            gemv_fp8_rs(&engine.shards[eg_w.shard_index as usize].mmap, eg_w.offset_start as usize,
                        &engine.shards[eg_s.shard_index as usize].mmap, eg_s.offset_start as usize,
                        &x_norm2, &mut gate_vec, hidden_dim, inter_dim);
            gemv_fp8_rs(&engine.shards[eu_w.shard_index as usize].mmap, eu_w.offset_start as usize,
                        &engine.shards[eu_s.shard_index as usize].mmap, eu_s.offset_start as usize,
                        &x_norm2, &mut up_vec, hidden_dim, inter_dim);

            for i in 0..inter_dim {
                let g = gate_vec[i];
                let silu_g = g / (1.0 + (-g).exp());
                inter_vec[i] = silu_g * up_vec[i];
            }

            gemv_fp8_rs(&engine.shards[ed_w.shard_index as usize].mmap, ed_w.offset_start as usize,
                        &engine.shards[ed_s.shard_index as usize].mmap, ed_s.offset_start as usize,
                        &inter_vec, &mut down_vec, inter_dim, hidden_dim);

            for d in 0..hidden_dim {
                h_mlp[d] += down_vec[d] * p_k;
            }
        }

        println!("h_mlp first 8 dims: {:?}", &h_mlp[0..8]);

        // 15. Residual 2 (h_next = h_mid + h_mlp)
        let mut h_next = vec![0.0f32; hidden_dim];
        for d in 0..hidden_dim {
            h_next[d] = h_mid[d] + h_mlp[d];
        }
        println!("h_next (Layer 0 output) first 8 dims: {:?}", &h_next[0..8]);
    }

    #[test]
    fn test_cpu_full_forward() {
        let snapshot_dir = PathBuf::from("/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c");
        let index_file = snapshot_dir.join("model.safetensors.index.json");
        if !index_file.exists() { return; }

        let engine = DynaMoeEngine::new(index_file.to_string_lossy().to_string()).unwrap();
        let summary = engine.get_summary().unwrap();

        let tok_path = snapshot_dir.join("tokenizer.json");
        let tokenizer = DynaMoeTokenizer::new(tok_path.to_string_lossy().to_string()).unwrap();

        let hidden_dim = 2048usize;
        let vocab_size = 248320usize;
        let num_layers = 40usize;

        // RoPE tables for 256 dim (rotary dim = 64)
        let rotary_dim = 64usize;
        let head_dim_full = 256usize;
        let rope_theta = 1000000.0f32;
        let mut cos_table = vec![0.0f32; rotary_dim / 2];
        let mut sin_table = vec![0.0f32; rotary_dim / 2];
        let pos = 0usize; // step 0
        for i in 0..(rotary_dim / 2) {
            let freq = 1.0 / (rope_theta.powf((2 * i) as f32 / rotary_dim as f32));
            let angle = (pos as f32) * freq;
            cos_table[i] = angle.cos();
            sin_table[i] = angle.sin();
        }

        let embed_tensor = summary.tensors.iter().find(|t| t.name == "model.language_model.embed_tokens.weight").unwrap();
        let embed_mmap = &engine.shards[embed_tensor.shard_index as usize].mmap;
        let embed_u16 = unsafe {
            std::slice::from_raw_parts(embed_mmap.as_ptr().add(embed_tensor.offset_start as usize) as *const u16, vocab_size * hidden_dim)
        };

        let target_token_id: usize = 248045; // <|im_start|>
        let mut h = vec![0.0f32; hidden_dim];
        for d in 0..hidden_dim {
            h[d] = bf16_to_f32(embed_u16[target_token_id * hidden_dim + d]);
        }

        // KV cache & conv / linear state
        let mut conv_states = vec![0.0f32; 30 * 8192 * 4];
        let mut linear_states = vec![0.0f32; 30 * 32 * 128 * 128];
        let mut kv_k_cache = vec![0.0f32; 10 * 2048 * 2 * 256];
        let mut kv_v_cache = vec![0.0f32; 10 * 2048 * 2 * 256];

        let mut lin_layer_count = 0usize;
        let mut full_layer_count = 0usize;

        for l in 0..num_layers {
            let is_full = (l % 4 == 3);

            // 1. Input layernorm
            let norm1_name = format!("model.language_model.layers.{}.input_layernorm.weight", l);
            let norm1 = summary.tensors.iter().find(|t| t.name == norm1_name).unwrap();
            let mut x_norm1 = vec![0.0f32; hidden_dim];
            rmsnorm_rs(&h, &engine.shards[norm1.shard_index as usize].mmap, norm1.offset_start as usize, &mut x_norm1, 1e-6);

            let mut attn_out = vec![0.0f32; hidden_dim];

            if is_full {
                let full_idx = full_layer_count;
                full_layer_count += 1;

                // Full Attention (GQA)
                let q_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.q_proj.weight", l)).unwrap();
                let q_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.q_proj.weight_scale", l)).unwrap();
                let k_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.k_proj.weight", l)).unwrap();
                let k_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.k_proj.weight_scale", l)).unwrap();
                let v_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.v_proj.weight", l)).unwrap();
                let v_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.v_proj.weight_scale", l)).unwrap();
                let q_norm = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.q_norm.weight", l)).unwrap();
                let k_norm = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.k_norm.weight", l)).unwrap();
                let o_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.o_proj.weight", l)).unwrap();
                let o_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.self_attn.o_proj.weight_scale", l)).unwrap();

                let mut q_raw = vec![0.0f32; 8192];
                let mut k_raw = vec![0.0f32; 512];
                let mut v_raw = vec![0.0f32; 512];

                gemv_fp8_rs(&engine.shards[q_w.shard_index as usize].mmap, q_w.offset_start as usize,
                            &engine.shards[q_s.shard_index as usize].mmap, q_s.offset_start as usize,
                            &x_norm1, &mut q_raw, hidden_dim, 8192);
                gemv_fp8_rs(&engine.shards[k_w.shard_index as usize].mmap, k_w.offset_start as usize,
                            &engine.shards[k_s.shard_index as usize].mmap, k_s.offset_start as usize,
                            &x_norm1, &mut k_raw, hidden_dim, 512);
                gemv_fp8_rs(&engine.shards[v_w.shard_index as usize].mmap, v_w.offset_start as usize,
                            &engine.shards[v_s.shard_index as usize].mmap, v_s.offset_start as usize,
                            &x_norm1, &mut v_raw, hidden_dim, 512);

                // q_norm and k_norm
                let qn_u16 = unsafe {
                    std::slice::from_raw_parts(engine.shards[q_norm.shard_index as usize].mmap.as_ptr().add(q_norm.offset_start as usize) as *const u16, 256)
                };
                let kn_u16 = unsafe {
                    std::slice::from_raw_parts(engine.shards[k_norm.shard_index as usize].mmap.as_ptr().add(k_norm.offset_start as usize) as *const u16, 256)
                };

                let mut q_vec = vec![0.0f32; 16 * 256];
                let mut gate_vec = vec![0.0f32; 16 * 256];

                for h_idx in 0..16 {
                    let h_start = h_idx * 512;
                    let mut sum_sq = 0.0f32;
                    for d in 0..256 {
                        let q_val = q_raw[h_start + d];
                        sum_sq += q_val * q_val;
                    }
                    let inv_rms = 1.0 / ((sum_sq / 256.0) + 1e-6).sqrt();
                    for d in 0..256 {
                        q_vec[h_idx * 256 + d] = q_raw[h_start + d] * inv_rms * (1.0 + bf16_to_f32(qn_u16[d]));
                        gate_vec[h_idx * 256 + d] = q_raw[h_start + 256 + d];
                    }

                    // RoPE on Q (rotary dim 64)
                    for i in 0..(rotary_dim / 2) {
                        let q0 = q_vec[h_idx * 256 + i];
                        let q1 = q_vec[h_idx * 256 + i + 32];
                        let cos = cos_table[i];
                        let sin = sin_table[i];
                        q_vec[h_idx * 256 + i] = q0 * cos - q1 * sin;
                        q_vec[h_idx * 256 + i + 32] = q0 * sin + q1 * cos;
                    }
                }

                let mut k_vec = vec![0.0f32; 2 * 256];
                for kv_h in 0..2 {
                    let mut sum_sq = 0.0f32;
                    for d in 0..256 {
                        let k_val = k_raw[kv_h * 256 + d];
                        sum_sq += k_val * k_val;
                    }
                    let inv_rms = 1.0 / ((sum_sq / 256.0) + 1e-6).sqrt();
                    for d in 0..256 {
                        k_vec[kv_h * 256 + d] = k_raw[kv_h * 256 + d] * inv_rms * (1.0 + bf16_to_f32(kn_u16[d]));
                    }
                    // RoPE on K
                    for i in 0..(rotary_dim / 2) {
                        let k0 = k_vec[kv_h * 256 + i];
                        let k1 = k_vec[kv_h * 256 + i + 32];
                        let cos = cos_table[i];
                        let sin = sin_table[i];
                        k_vec[kv_h * 256 + i] = k0 * cos - k1 * sin;
                        k_vec[kv_h * 256 + i + 32] = k0 * sin + k1 * cos;
                    }
                }

                // Store in KV cache (pos = 0)
                let kv_base = full_idx * 2048 * 2 * 256;
                for i in 0..(2 * 256) {
                    kv_k_cache[kv_base + i] = k_vec[i];
                    kv_v_cache[kv_base + i] = v_raw[i];
                }

                // Step 0 GQA (single token attending to itself)
                let mut ctx = vec![0.0f32; 16 * 256];
                let scale = 1.0 / (256.0f32).sqrt();

                for h_idx in 0..16 {
                    let kv_h = h_idx / 8; // 16 heads / 2 kv heads = 8
                    let mut dot = 0.0f32;
                    for d in 0..256 {
                        dot += q_vec[h_idx * 256 + d] * k_vec[kv_h * 256 + d];
                    }
                    // softmax of single element is 1.0
                    for d in 0..256 {
                        let v_val = v_raw[kv_h * 256 + d];
                        let g = gate_vec[h_idx * 256 + d];
                        let silu_g = 1.0 / (1.0 + (-g).exp());
                        ctx[h_idx * 256 + d] = v_val * silu_g;
                    }
                }

                // o_proj (2048 x 4096)
                gemv_fp8_rs(&engine.shards[o_w.shard_index as usize].mmap, o_w.offset_start as usize,
                            &engine.shards[o_s.shard_index as usize].mmap, o_s.offset_start as usize,
                            &ctx, &mut attn_out, 4096, hidden_dim);
            } else {
                let lin_idx = lin_layer_count;
                lin_layer_count += 1;

                // Linear Attention
                let in_qkv = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.in_proj_qkv.weight", l)).unwrap();
                let conv_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.conv1d.weight", l)).unwrap();
                let in_z = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.in_proj_z.weight", l)).unwrap();
                let in_a = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.in_proj_a.weight", l)).unwrap();
                let in_b = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.in_proj_b.weight", l)).unwrap();
                let a_log_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.A_log", l)).unwrap();
                let dt_bias_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.dt_bias", l)).unwrap();
                let lin_norm_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.norm.weight", l)).unwrap();
                let out_proj_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.linear_attn.out_proj.weight", l)).unwrap();

                let mut qkv = vec![0.0f32; 8192];
                gemv_bf16_rs(&engine.shards[in_qkv.shard_index as usize].mmap, in_qkv.offset_start as usize, &x_norm1, &mut qkv, hidden_dim, 8192);

                // conv1d
                let conv_u16 = unsafe {
                    std::slice::from_raw_parts(engine.shards[conv_t.shard_index as usize].mmap.as_ptr().add(conv_t.offset_start as usize) as *const u16, 8192 * 4)
                };
                let mut conv_out = vec![0.0f32; 8192];
                for c in 0..8192 {
                    let w3 = bf16_to_f32(conv_u16[c * 4 + 3]);
                    let conv_val = w3 * qkv[c];
                    conv_out[c] = conv_val / (1.0 + (-conv_val).exp());
                }

                // L2-norm
                for h in 0..16 {
                    let q_base = h * 128;
                    let mut q_sum_sq = 0.0f32;
                    for i in 0..128 { q_sum_sq += conv_out[q_base + i] * conv_out[q_base + i]; }
                    let inv_q = 1.0 / (q_sum_sq + 1e-6).sqrt();
                    for i in 0..128 { conv_out[q_base + i] *= inv_q; }

                    let k_base = 2048 + (h * 128);
                    let mut k_sum_sq = 0.0f32;
                    for i in 0..128 { k_sum_sq += conv_out[k_base + i] * conv_out[k_base + i]; }
                    let inv_k = 1.0 / (k_sum_sq + 1e-6).sqrt();
                    for i in 0..128 { conv_out[k_base + i] *= inv_k; }
                }

                let mut z = vec![0.0f32; 4096];
                gemv_bf16_rs(&engine.shards[in_z.shard_index as usize].mmap, in_z.offset_start as usize, &x_norm1, &mut z, hidden_dim, 4096);

                let mut a_vec = vec![0.0f32; 32];
                gemv_bf16_rs(&engine.shards[in_a.shard_index as usize].mmap, in_a.offset_start as usize, &x_norm1, &mut a_vec, hidden_dim, 32);

                let mut b_vec = vec![0.0f32; 32];
                gemv_bf16_rs(&engine.shards[in_b.shard_index as usize].mmap, in_b.offset_start as usize, &x_norm1, &mut b_vec, hidden_dim, 32);

                let a_log_u16 = unsafe {
                    std::slice::from_raw_parts(engine.shards[a_log_t.shard_index as usize].mmap.as_ptr().add(a_log_t.offset_start as usize) as *const u16, 32)
                };
                let dt_bias_u16 = unsafe {
                    std::slice::from_raw_parts(engine.shards[dt_bias_t.shard_index as usize].mmap.as_ptr().add(dt_bias_t.offset_start as usize) as *const u16, 32)
                };
                let lin_norm_u16 = unsafe {
                    std::slice::from_raw_parts(engine.shards[lin_norm_t.shard_index as usize].mmap.as_ptr().add(lin_norm_t.offset_start as usize) as *const u16, 128)
                };

                let mut attn_ctx = vec![0.0f32; 4096];
                let lin_state_base = lin_idx * (32 * 128 * 128);

                for h_idx in 0..32 {
                    let key_h = h_idx / 2;
                    let q_base = key_h * 128;
                    let k_base = 2048 + (key_h * 128);
                    let v_base = 4096 + (h_idx * 128);
                    let z_base = h_idx * 128;
                    let out_base = h_idx * 128;
                    let state_base = lin_state_base + h_idx * 128 * 128;

                    let a_log_val = bf16_to_f32(a_log_u16[h_idx]);
                    let dt_bias_val = bf16_to_f32(dt_bias_u16[h_idx]);
                    let a_val = a_vec[h_idx];
                    let b_val = b_vec[h_idx];

                    let x = a_val + dt_bias_val;
                    let dt = if x > 20.0 { x } else if x < -20.0 { x.exp() } else { (1.0 + x.exp()).ln() };
                    let alpha = (-a_log_val.exp() * dt).exp();
                    let beta = 1.0 / (1.0 + (-b_val).exp());

                    let mut sk = [0.0f32; 128];
                    for i in 0..128 {
                        let mut sum = 0.0f32;
                        for j in 0..128 {
                            sum += linear_states[state_base + (i * 128) + j] * conv_out[k_base + j];
                        }
                        sk[i] = sum;
                    }

                    let mut y = [0.0f32; 128];
                    let mut sum_sq = 0.0f32;
                    for i in 0..128 {
                        let vi = conv_out[v_base + i];
                        let ui = vi - (alpha * sk[i]);
                        let beta_ui = beta * ui;

                        let mut yi = 0.0f32;
                        for j in 0..128 {
                            let s_idx = state_base + (i * 128) + j;
                            let s_old = linear_states[s_idx];
                            let kj = conv_out[k_base + j];
                            let s_new = (alpha * s_old) + (beta_ui * kj);
                            linear_states[s_idx] = s_new;
                            yi += s_new * conv_out[q_base + j];
                        }
                        y[i] = yi;
                        sum_sq += yi * yi;
                    }

                    let inv_rms = 1.0 / ((sum_sq / 128.0) + 1e-6).sqrt();
                    for i in 0..128 {
                        let norm_w = bf16_to_f32(lin_norm_u16[i]);
                        let y_norm = y[i] * inv_rms * norm_w;
                        let z_val = z[z_base + i];
                        let silu_z = z_val / (1.0 + (-z_val).exp());
                        attn_ctx[out_base + i] = y_norm * silu_z;
                    }
                }

                gemv_bf16_rs(&engine.shards[out_proj_t.shard_index as usize].mmap, out_proj_t.offset_start as usize, &attn_ctx, &mut attn_out, 4096, hidden_dim);
            }

            // Residual 1
            let mut h_mid = vec![0.0f32; hidden_dim];
            for d in 0..hidden_dim {
                h_mid[d] = h[d] + attn_out[d];
            }

            // Post attention norm
            let norm2 = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.post_attention_layernorm.weight", l)).unwrap();
            let mut x_norm2 = vec![0.0f32; hidden_dim];
            rmsnorm_rs(&h_mid, &engine.shards[norm2.shard_index as usize].mmap, norm2.offset_start as usize, &mut x_norm2, 1e-6);

            // Router
            let router_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.gate.weight", l)).unwrap();
            let mut router_logits = vec![0.0f32; 256];
            gemv_bf16_rs(&engine.shards[router_t.shard_index as usize].mmap, router_t.offset_start as usize, &x_norm2, &mut router_logits, hidden_dim, 256);

            let mut router_pairs: Vec<(usize, f32)> = router_logits.iter().cloned().enumerate().collect();
            router_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top8: Vec<(usize, f32)> = router_pairs.into_iter().take(8).collect();
            let max_l = top8[0].1;
            let exp_sum: f32 = top8.iter().map(|(_, l)| (l - max_l).exp()).sum();
            let top8_weights: Vec<(usize, f32)> = top8.iter().map(|(id, l)| (*id, (l - max_l).exp() / exp_sum)).collect();

            // Shared expert gate
            let shared_gate_t = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert_gate.weight", l)).unwrap();
            let shared_gate_u16 = unsafe {
                std::slice::from_raw_parts(engine.shards[shared_gate_t.shard_index as usize].mmap.as_ptr().add(shared_gate_t.offset_start as usize) as *const u16, hidden_dim)
            };
            let mut sg_dot = 0.0f32;
            for d in 0..hidden_dim {
                sg_dot += bf16_to_f32(shared_gate_u16[d]) * x_norm2[d];
            }
            let shared_w = 1.0 / (1.0 + (-sg_dot).exp());

            // MLP
            let mut h_mlp = vec![0.0f32; hidden_dim];
            let inter_dim = 512usize;

            // Shared expert
            let sg_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert.gate_proj.weight", l)).unwrap();
            let sg_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert.gate_proj.weight_scale", l)).unwrap();
            let su_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert.up_proj.weight", l)).unwrap();
            let su_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert.up_proj.weight_scale", l)).unwrap();
            let sd_w = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert.down_proj.weight", l)).unwrap();
            let sd_s = summary.tensors.iter().find(|t| t.name == format!("model.language_model.layers.{}.mlp.shared_expert.down_proj.weight_scale", l)).unwrap();

            let mut gate_vec = vec![0.0f32; inter_dim];
            let mut up_vec = vec![0.0f32; inter_dim];
            let mut inter_vec = vec![0.0f32; inter_dim];
            let mut down_vec = vec![0.0f32; hidden_dim];

            gemv_fp8_rs(&engine.shards[sg_w.shard_index as usize].mmap, sg_w.offset_start as usize,
                        &engine.shards[sg_s.shard_index as usize].mmap, sg_s.offset_start as usize,
                        &x_norm2, &mut gate_vec, hidden_dim, inter_dim);
            gemv_fp8_rs(&engine.shards[su_w.shard_index as usize].mmap, su_w.offset_start as usize,
                        &engine.shards[su_s.shard_index as usize].mmap, su_s.offset_start as usize,
                        &x_norm2, &mut up_vec, hidden_dim, inter_dim);

            for i in 0..inter_dim {
                let g = gate_vec[i];
                inter_vec[i] = (g / (1.0 + (-g).exp())) * up_vec[i];
            }

            gemv_fp8_rs(&engine.shards[sd_w.shard_index as usize].mmap, sd_w.offset_start as usize,
                        &engine.shards[sd_s.shard_index as usize].mmap, sd_s.offset_start as usize,
                        &inter_vec, &mut down_vec, inter_dim, hidden_dim);

            for d in 0..hidden_dim {
                h_mlp[d] += down_vec[d] * shared_w;
            }

            // Top 8 routed experts
            for (exp_id, p_k) in top8_weights {
                let eg_w_name = format!("model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight", l, exp_id);
                let eg_s_name = format!("model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight_scale", l, exp_id);
                let eu_w_name = format!("model.language_model.layers.{}.mlp.experts.{}.up_proj.weight", l, exp_id);
                let eu_s_name = format!("model.language_model.layers.{}.mlp.experts.{}.up_proj.weight_scale", l, exp_id);
                let ed_w_name = format!("model.language_model.layers.{}.mlp.experts.{}.down_proj.weight", l, exp_id);
                let ed_s_name = format!("model.language_model.layers.{}.mlp.experts.{}.down_proj.weight_scale", l, exp_id);

                let eg_w = summary.tensors.iter().find(|t| t.name == eg_w_name).unwrap();
                let eg_s = summary.tensors.iter().find(|t| t.name == eg_s_name).unwrap();
                let eu_w = summary.tensors.iter().find(|t| t.name == eu_w_name).unwrap();
                let eu_s = summary.tensors.iter().find(|t| t.name == eu_s_name).unwrap();
                let ed_w = summary.tensors.iter().find(|t| t.name == ed_w_name).unwrap();
                let ed_s = summary.tensors.iter().find(|t| t.name == ed_s_name).unwrap();

                gemv_fp8_rs(&engine.shards[eg_w.shard_index as usize].mmap, eg_w.offset_start as usize,
                            &engine.shards[eg_s.shard_index as usize].mmap, eg_s.offset_start as usize,
                            &x_norm2, &mut gate_vec, hidden_dim, inter_dim);
                gemv_fp8_rs(&engine.shards[eu_w.shard_index as usize].mmap, eu_w.offset_start as usize,
                            &engine.shards[eu_s.shard_index as usize].mmap, eu_s.offset_start as usize,
                            &x_norm2, &mut up_vec, hidden_dim, inter_dim);

                for i in 0..inter_dim {
                    let g = gate_vec[i];
                    inter_vec[i] = (g / (1.0 + (-g).exp())) * up_vec[i];
                }

                gemv_fp8_rs(&engine.shards[ed_w.shard_index as usize].mmap, ed_w.offset_start as usize,
                            &engine.shards[ed_s.shard_index as usize].mmap, ed_s.offset_start as usize,
                            &inter_vec, &mut down_vec, inter_dim, hidden_dim);

                for d in 0..hidden_dim {
                    h_mlp[d] += down_vec[d] * p_k;
                }
            }

            // Residual 2
            for d in 0..hidden_dim {
                h[d] = h_mid[d] + h_mlp[d];
            }

            if l % 5 == 0 || l == 39 {
                println!("Layer {} output h[0..4]: {:?}", l, &h[0..4]);
            }
        }

        // Final RMSNorm
        let final_norm = summary.tensors.iter().find(|t| t.name == "model.language_model.norm.weight").unwrap();
        let mut x_final = vec![0.0f32; hidden_dim];
        rmsnorm_rs(&h, &engine.shards[final_norm.shard_index as usize].mmap, final_norm.offset_start as usize, &mut x_final, 1e-6);
        println!("x_final first 8 dims: {:?}", &x_final[0..8]);

        // LM Head
        let lm_head = summary.tensors.iter().find(|t| t.name == "lm_head.weight").unwrap();
        let mut logits = vec![0.0f32; vocab_size];
        gemv_bf16_rs(&engine.shards[lm_head.shard_index as usize].mmap, lm_head.offset_start as usize, &x_final, &mut logits, hidden_dim, vocab_size);

        let mut logit_pairs: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        logit_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("=== TOP 10 PREDICTED TOKENS (from prompt <|im_start|>) ===");
        for (tok_id, val) in logit_pairs.into_iter().take(10) {
            let piece = tokenizer.decode(vec![tok_id as u32]).unwrap_or_default();
            println!("  Token {}: logit={:.3}, text={:?}", tok_id, val, piece);
        }
    }

    #[test]
    fn test_real_mlx_4bit_path() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/derekparris".to_string());
        let snapshot_dir = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-MLX-4bit/snapshots/19504d912fa8fc7622bf6b1de3db5d5d890b1f02");

        if !snapshot_dir.exists() {
            println!("Skipping test_real_mlx_4bit_path because snapshot directory is not present.");
            return;
        }

        let engine = DynaMoeEngine::new(snapshot_dir.to_string_lossy().to_string()).expect("Failed to initialize engine for MLX 4-bit");
        let summary = engine.get_summary().expect("Failed to get summary");

        println!("Loaded MLX 4-bit Model Summary: size={:.2} GB, total_tensors={}, layers={}", summary.size_gb, summary.tensor_count, summary.layers.len());
        assert_eq!(summary.layers.len(), 40, "Expected 40 layers in MLX model");

        // Verify that every layer has 256 routed experts unbundled
        for l in &summary.layers {
            assert_eq!(l.routed_expert_count, 256, "Layer {} should have 256 routed experts", l.layer_index);
        }

        // Verify that router gate and shared expert gate are correctly identified
        let router_0 = summary.tensors.iter().find(|t| t.name == "language_model.model.layers.0.mlp.gate.weight").unwrap();
        assert_eq!(router_0.category, "MoE Router");

        let shared_gate_0 = summary.tensors.iter().find(|t| t.name == "language_model.model.layers.0.mlp.shared_expert_gate.weight").unwrap();
        assert_eq!(shared_gate_0.category, "Shared Expert Gate");

        let exp0_gate_w = summary.tensors.iter().find(|t| t.name == "language_model.model.layers.0.mlp.experts.0.gate_proj.weight").unwrap();
        assert_eq!(exp0_gate_w.category, "Routed Expert #0");
        assert_eq!(exp0_gate_w.shape_display, "[512, 256]");
        assert_eq!(exp0_gate_w.expert_id, Some(0));

        let exp255_down_w = summary.tensors.iter().find(|t| t.name == "language_model.model.layers.0.mlp.experts.255.down_proj.weight").unwrap();
        assert_eq!(exp255_down_w.category, "Routed Expert #255");
        assert_eq!(exp255_down_w.shape_display, "[2048, 64]");
        assert_eq!(exp255_down_w.expert_id, Some(255));
    }
}