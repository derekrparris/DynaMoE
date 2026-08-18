uniffi::setup_scaffolding!();

use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use std::fmt;
use std::fs::File;
use std::sync::Arc;

#[derive(Debug, uniffi::Error)]
pub enum EngineError {
    FileError { details: String },
    MmapError { details: String },
    ParseError { details: String },
    TensorNotFound { name: String },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::FileError { details } => write!(f, "File error: {}", details),
            EngineError::MmapError { details } => write!(f, "Mmap error: {}", details),
            EngineError::ParseError { details } => write!(f, "Parse error: {}", details),
            EngineError::TensorNotFound { name } => write!(f, "Tensor not found: {}", name),
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
}

#[derive(uniffi::Record)]
pub struct ModelSummary {
    pub size_gb: f64,
    pub tensor_count: u32,
    pub tensors: Vec<TensorMetadata>,
}

#[derive(uniffi::Object)]
pub struct DynaMoeEngine {
    mmap: Mmap,
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

        for name in tensors.names() {
            if let Ok(tensor) = tensors.tensor(name) {
                // Get header offset and data slice relative to mmap base
                let data_ptr = tensor.data().as_ptr() as usize;
                let base_ptr = self.mmap.as_ptr() as usize;
                
                let offset_start = (data_ptr - base_ptr) as u64;
                let offset_end = offset_start + tensor.data().len() as u64;

                tensor_list.push(TensorMetadata {
                    name: name.to_string(),
                    shape_display: format!("{:?}", tensor.shape()),
                    dtype: format!("{:?}", tensor.dtype()),
                    size_mb: tensor.data().len() as f64 / (1024.0 * 1024.0),
                    offset_start,
                    offset_end,
                });
            }
        }

        tensor_list.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(ModelSummary {
            size_gb,
            tensor_count: tensors.names().len() as u32,
            tensors: tensor_list,
        })
    }

    pub fn buffer_base_address(&self) -> u64 {
        self.mmap.as_ptr() as u64
    }

    pub fn buffer_length(&self) -> u64 {
        self.mmap.len() as u64
    }
}