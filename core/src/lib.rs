// This macro sets up the internal C-FFI boundary
uniffi::setup_scaffolding!();

// The #[uniffi::export] tag tells UniFFI to generate Swift code for this function
#[uniffi::export]
pub fn hello_from_dynamoe(name: String) -> String {
    format!("Hello, {}! The DynaMoE Rust engine is awake and ready.", name)
}