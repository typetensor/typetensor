/*!
 * Type definitions for WebAssembly backend
 * 
 * Defines the core data types and enums used throughout the WASM backend,
 * with a focus on efficient memory representation and JS interoperability.
 */

use wasm_bindgen::prelude::*;

/// Data type enumeration matching TypeTensor's core dtype system
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmDType {
    Int8 = 0,
    Uint8 = 1,
    Int16 = 2,
    Uint16 = 3,
    Int32 = 4,
    Uint32 = 5,
    Float32 = 6,
    Float64 = 7,
    BigInt64 = 8,
    BigUint64 = 9,
}

impl WasmDType {
    /// Get the byte size of this data type
    pub fn byte_size(&self) -> usize {
        match self {
            WasmDType::Int8 | WasmDType::Uint8 => 1,
            WasmDType::Int16 | WasmDType::Uint16 => 2,
            WasmDType::Int32 | WasmDType::Uint32 | WasmDType::Float32 => 4,
            WasmDType::Float64 | WasmDType::BigInt64 | WasmDType::BigUint64 => 8,
        }
    }

    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, WasmDType::Float32 | WasmDType::Float64)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        !self.is_float()
    }

    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        matches!(self, WasmDType::Int8 | WasmDType::Int16 | WasmDType::Int32 | WasmDType::BigInt64 | WasmDType::Float32 | WasmDType::Float64)
    }

    /// Check if this is a BigInt type
    pub fn is_bigint(&self) -> bool {
        matches!(self, WasmDType::BigInt64 | WasmDType::BigUint64)
    }
}

/// Tensor operation types matching TypeTensor's core operation system
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmOperation {
    // Creation
    Create = 0,
    
    // Unary operations
    Neg = 1,
    Abs = 2,
    Sin = 3,
    Cos = 4,
    Exp = 5,
    Log = 6,
    Sqrt = 7,
    Square = 8,
    
    // Binary operations
    Add = 10,
    Sub = 11,
    Mul = 12,
    Div = 13,
    
    // View operations
    Reshape = 20,
    View = 21,
    Slice = 22,
    Flatten = 23,
    Permute = 24,
    Transpose = 25,
    Squeeze = 26,
    Unsqueeze = 27,
    Expand = 28,
    Tile = 29,
    
    // Matrix operations
    Matmul = 30,
    
    // Activation functions
    Softmax = 40,
    LogSoftmax = 41,
    
    // Reduction operations
    Sum = 50,
    Mean = 51,
    Max = 52,
    Min = 53,
    Prod = 54,
    
    // Einops operations
    Rearrange = 60,
    Reduce = 61,
}

/// Memory layout flags for tensors
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmLayout {
    pub c_contiguous: bool,
    pub f_contiguous: bool,
    pub is_view: bool,
    pub writeable: bool,
    pub aligned: bool,
}

impl Default for WasmLayout {
    fn default() -> Self {
        WasmLayout {
            c_contiguous: true,
            f_contiguous: false,
            is_view: false,
            writeable: true,
            aligned: true,
        }
    }
}

/// Tensor metadata structure
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmTensorMeta {
    dtype: WasmDType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
    layout: WasmLayout,
    offset: usize,
}

#[wasm_bindgen]
impl WasmTensorMeta {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dtype: WasmDType,
        shape: Vec<usize>,
        strides: Vec<usize>,
        size: usize,
        offset: usize,
    ) -> WasmTensorMeta {
        WasmTensorMeta {
            dtype,
            shape,
            strides,
            size,
            layout: WasmLayout::default(),
            offset,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> WasmDType {
        self.dtype
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.size
    }

    #[wasm_bindgen(getter)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[wasm_bindgen(getter)]
    pub fn byte_size(&self) -> usize {
        self.size * self.dtype.byte_size()
    }
}

/// Error types for WASM operations
#[derive(Debug)]
pub enum WasmError {
    InvalidOperation,
    InvalidDType,
    InvalidShape,
    OutOfMemory,
    InvalidInput,
    NotImplemented,
    MemoryAllocationFailed,
}

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        let message = match error {
            WasmError::InvalidOperation => "Invalid operation",
            WasmError::InvalidDType => "Invalid data type",
            WasmError::InvalidShape => "Invalid tensor shape",
            WasmError::OutOfMemory => "Out of memory",
            WasmError::InvalidInput => "Invalid input",
            WasmError::NotImplemented => "Operation not yet implemented",
            WasmError::MemoryAllocationFailed => "Memory allocation failed",
        };
        js_sys::Error::new(message).into()
    }
}

/// Result type for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_byte_sizes() {
        assert_eq!(WasmDType::Int8.byte_size(), 1);
        assert_eq!(WasmDType::Uint8.byte_size(), 1);
        assert_eq!(WasmDType::Int16.byte_size(), 2);
        assert_eq!(WasmDType::Uint16.byte_size(), 2);
        assert_eq!(WasmDType::Int32.byte_size(), 4);
        assert_eq!(WasmDType::Uint32.byte_size(), 4);
        assert_eq!(WasmDType::Float32.byte_size(), 4);
        assert_eq!(WasmDType::Float64.byte_size(), 8);
        assert_eq!(WasmDType::BigInt64.byte_size(), 8);
        assert_eq!(WasmDType::BigUint64.byte_size(), 8);
    }

    #[test]
    fn test_dtype_properties() {
        assert!(WasmDType::Float32.is_float());
        assert!(!WasmDType::Int32.is_float());
        assert!(WasmDType::Int32.is_signed());
        assert!(!WasmDType::Uint32.is_signed());
        assert!(WasmDType::BigInt64.is_bigint());
        assert!(!WasmDType::Float32.is_bigint());
    }

    #[test]
    fn test_tensor_meta() {
        let meta = WasmTensorMeta::new(
            WasmDType::Float32,
            vec![2, 3, 4],
            vec![12, 4, 1],
            24,
            0,
        );
        
        assert_eq!(meta.dtype(), WasmDType::Float32);
        assert_eq!(meta.shape(), vec![2, 3, 4]);
        assert_eq!(meta.size(), 24);
        assert_eq!(meta.byte_size(), 96); // 24 * 4 bytes
    }
}