uniffi::setup_scaffolding!();

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;

#[derive(uniffi::Record)]
pub struct TensorMetadata {
    pub name: String,
    pub shape_display: String,
    pub dtype: String,
    pub size_mb: f64,
}

#[derive(uniffi::Record)]
pub struct ModelSummary {
    pub file_path: String,
    pub size_gb: f64,
    pub tensor_count: u32,
    pub tensors: Vec<TensorMetadata>,
    pub error: Option<String>,
}

#[uniffi::export]
pub fn inspect_model_weights(file_path: String) -> ModelSummary {
    let file = match File::open(&file_path) {
        Ok(f) => f,
        Err(e) => {
            return ModelSummary {
                file_path,
                size_gb: 0.0,
                tensor_count: 0,
                tensors: Vec::new(),
                error: Some(format!("Could not open file: {}", e)),
            }
        }
    };

    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            return ModelSummary {
                file_path,
                size_gb: 0.0,
                tensor_count: 0,
                tensors: Vec::new(),
                error: Some(format!("Could not memory-map file: {}", e)),
            }
        }
    };

    let tensors = match SafeTensors::deserialize(&mmap) {
        Ok(t) => t,
        Err(e) => {
            return ModelSummary {
                file_path,
                size_gb: 0.0,
                tensor_count: 0,
                tensors: Vec::new(),
                error: Some(format!("Invalid safetensors format: {:?}", e)),
            }
        }
    };

    let size_gb = mmap.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let tensor_count = tensors.names().len() as u32;

    let mut tensor_list = Vec::new();

    for name in tensors.names() {
        if let Ok(tensor) = tensors.tensor(name) {
            let shape_str = format!("{:?}", tensor.shape());
            let dtype_str = format!("{:?}", tensor.dtype());
            let size_mb = tensor.data().len() as f64 / (1024.0 * 1024.0);

            tensor_list.push(TensorMetadata {
                name: name.to_string(),
                shape_display: shape_str,
                dtype: dtype_str,
                size_mb,
            });
        }
    }

    // Sort matrices alphabetically for clean UI presentation
    tensor_list.sort_by(|a, b| a.name.cmp(&b.name));

    ModelSummary {
        file_path,
        size_gb,
        tensor_count,
        tensors: tensor_list,
        error: None,
    }
}