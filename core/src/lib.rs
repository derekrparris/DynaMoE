uniffi::setup_scaffolding!();

use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::sync::Arc;

#[derive(Debug, uniffi::Error)]
pub enum EngineError {
    FileError { details: String },
    MmapError { details: String },
    ParseError { details: String },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::FileError { details } => write!(f, "File error: {}", details),
            EngineError::MmapError { details } => write!(f, "Mmap error: {}", details),
            EngineError::ParseError { details } => write!(f, "Parse error: {}", details),
        }
    }
}

#[derive(uniffi::Record)]
pub struct TensorMetadata {
    pub name: String,
    pub shape_display: String,
    pub dtype: String,
    pub size_mb: f64,
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
    pub tensors: Vec<TensorMetadata>,
    pub layers: Vec<LayerSummary>,
}

#[derive(uniffi::Object)]
pub struct DynaMoeEngine {
    mmap: Mmap,
}

/// Helper: Parses tensor naming conventions across Qwen, Nanbeige, DeepSeek, and Llama MoE architectures
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

    let category = if name.contains("embed_tokens") {
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

#[uniffi::export]
impl DynaMoeEngine {
    #[uniffi::constructor]
    pub fn new(file_path: String) -> Result<Arc<Self>, EngineError> {
        let file = File::open(&file_path).map_err(|e| EngineError::FileError {
            details: e.to_string(),
        })?;

        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| EngineError::MmapError {
            details: e.to_string(),
        })?;

        Ok(Arc::new(Self { mmap }))
    }

    pub fn get_summary(&self) -> Result<ModelSummary, EngineError> {
        let tensors = SafeTensors::deserialize(&self.mmap).map_err(|e| EngineError::ParseError {
            details: format!("{:?}", e),
        })?;

        let size_gb = self.mmap.len() as f64 / (1024.0 * 1024.0 * 1024.0);
        let mut tensor_list = Vec::new();
        let mut layer_map: BTreeMap<u32, (u32, BTreeMap<u32, bool>, f64)> = BTreeMap::new();
        let mut max_expert_id = 0u32;

        for name in tensors.names() {
            if let Ok(tensor) = tensors.tensor(name) {
                let data_ptr = tensor.data().as_ptr() as usize;
                let base_ptr = self.mmap.as_ptr() as usize;
                
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
                    offset_start,
                    offset_end,
                    category,
                    layer_index,
                    expert_id,
                });
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

        let layer_count = layer_summaries.len() as u32;

        Ok(ModelSummary {
            size_gb,
            tensor_count: tensors.names().len() as u32,
            layer_count,
            max_expert_id,
            tensors: tensor_list,
            layers: layer_summaries,
        })
    }

    pub fn buffer_base_address(&self) -> u64 {
        self.mmap.as_ptr() as u64
    }

    pub fn buffer_length(&self) -> u64 {
        self.mmap.len() as u64
    }
}