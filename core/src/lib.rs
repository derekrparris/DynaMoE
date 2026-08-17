uniffi::setup_scaffolding!();

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;

#[uniffi::export]
pub fn hello_from_dynamoe(name: String) -> String {
    format!("Hello, {}! The DynaMoE automated build is working!", name)
}

/// Memory-maps a Safetensors file and reads its metadata structure
#[uniffi::export]
pub fn inspect_model_weights(file_path: String) -> String {
    // 1. Open the file handle
    let file = match File::open(&file_path) {
        Ok(f) => f,
        Err(e) => return format!("Error: Could not open file.\nDetails: {}", e),
    };

    // 2. Ask the macOS kernel to memory-map the file. 
    // This is `unsafe` in Rust because another process *could* modify the file while we read it.
    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => return format!("Error: Could not memory-map file.\nDetails: {}", e),
    };

    // 3. Parse the Safetensors header to find the weight matrices
    let tensors = match SafeTensors::deserialize(&mmap) {
        Ok(t) => t,
        Err(e) => return format!("Error: Invalid safetensors format.\nDetails: {:?}", e),
    };

    // 4. Extract some basic info to prove it worked
    let tensor_count = tensors.names().len();
    let total_bytes = mmap.len();
    let size_in_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    format!(
        "✅ Successfully memory-mapped weights!\n\nFile: {}\nSize: {:.2} GB\nMatrices Found: {}", 
        file_path, size_in_gb, tensor_count
    )
}