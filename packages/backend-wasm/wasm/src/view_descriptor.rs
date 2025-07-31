/*!
 * View descriptor for zero-copy tensor views
 * 
 * Manages metadata for tensor views that share underlying buffer data
 * but have different shapes, strides, or offsets.
 */

use crate::types::{WasmDType, WasmTensorMeta};
use crate::memory::{WasmBufferHandle, BufferId, WasmMemoryManager};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Type of view operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewType {
    Base,      // Original tensor, owns the data
    Reshape,   // Different shape, same data order
    Transpose, // Swapped axes, different strides
    Permute,   // Arbitrary axis reordering
    Slice,     // Subset of data
    Expand,    // Broadcasting with zero strides
    Squeeze,   // Remove unit dimensions
    Unsqueeze, // Add unit dimensions
}

/// Descriptor for a tensor view
#[derive(Debug, Clone)]
pub struct WasmViewDescriptor {
    /// The underlying buffer handle (may be shared)
    pub buffer: Arc<WasmBufferHandle>,
    /// Buffer ID for tracking
    pub buffer_id: BufferId,
    /// Type of view
    pub view_type: ViewType,
    /// Tensor metadata (shape, strides, dtype)
    pub meta: WasmTensorMeta,
    /// Reference count for this view
    pub ref_count: Arc<AtomicUsize>,
    /// Parent view if this is derived
    pub parent: Option<Arc<WasmViewDescriptor>>,
}

impl WasmViewDescriptor {
    /// Create a new base view (owns the buffer)
    pub fn new_base(buffer: WasmBufferHandle, meta: WasmTensorMeta) -> Self {
        let buffer_id = buffer.id();
        WasmViewDescriptor {
            buffer: Arc::new(buffer),
            buffer_id,
            view_type: ViewType::Base,
            meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: None,
        }
    }

    /// Create a reshape view
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        // Verify total elements match
        let old_size = self.meta.size();
        let new_size: usize = new_shape.iter().product();
        
        if old_size != new_size {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                old_size, new_shape, new_size
            ));
        }

        // Check if we can do zero-copy reshape
        if !self.is_contiguous() {
            return Err("Cannot reshape non-contiguous tensor without copying".to_string());
        }

        // Calculate new strides for row-major layout
        let new_strides = calculate_strides(&new_shape);
        
        let new_meta = WasmTensorMeta::new(
            self.meta.dtype(),
            new_shape,
            new_strides,
            new_size,
            self.meta.offset(),
        );

        Ok(WasmViewDescriptor {
            buffer: Arc::clone(&self.buffer),
            buffer_id: self.buffer_id,
            view_type: ViewType::Reshape,
            meta: new_meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: Some(Arc::new(self.clone())),
        })
    }

    /// Create a transpose view (2D only for now)
    pub fn transpose(&self) -> Result<Self, String> {
        let shape = self.meta.shape();
        let strides = self.meta.strides();
        
        if shape.len() != 2 {
            return Err("Transpose currently only supports 2D tensors".to_string());
        }

        // Swap dimensions and strides
        let new_shape = vec![shape[1], shape[0]];
        let new_strides = vec![strides[1], strides[0]];
        
        let new_meta = WasmTensorMeta::new(
            self.meta.dtype(),
            new_shape,
            new_strides,
            self.meta.size(),
            self.meta.offset(),
        );

        Ok(WasmViewDescriptor {
            buffer: Arc::clone(&self.buffer),
            buffer_id: self.buffer_id,
            view_type: ViewType::Transpose,
            meta: new_meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: Some(Arc::new(self.clone())),
        })
    }

    /// Create a permute view
    pub fn permute(&self, axes: Vec<usize>) -> Result<Self, String> {
        let shape = self.meta.shape();
        let strides = self.meta.strides();
        
        if axes.len() != shape.len() {
            return Err("Permutation must have same number of axes as tensor".to_string());
        }

        // Verify valid permutation
        let mut seen = vec![false; axes.len()];
        for &axis in &axes {
            if axis >= axes.len() || seen[axis] {
                return Err("Invalid permutation axes".to_string());
            }
            seen[axis] = true;
        }

        // Reorder shape and strides according to axes
        let new_shape: Vec<usize> = axes.iter().map(|&i| shape[i]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&i| strides[i]).collect();
        
        let new_meta = WasmTensorMeta::new(
            self.meta.dtype(),
            new_shape,
            new_strides,
            self.meta.size(),
            self.meta.offset(),
        );

        Ok(WasmViewDescriptor {
            buffer: Arc::clone(&self.buffer),
            buffer_id: self.buffer_id,
            view_type: ViewType::Permute,
            meta: new_meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: Some(Arc::new(self.clone())),
        })
    }

    /// Create a squeeze view (remove dimensions of size 1)
    pub fn squeeze(&self, axes: Option<Vec<usize>>) -> Result<Self, String> {
        let shape = self.meta.shape();
        let strides = self.meta.strides();
        
        let axes_to_squeeze = match axes {
            Some(axes) => {
                // Validate specified axes
                for &axis in &axes {
                    if axis >= shape.len() {
                        return Err(format!("Axis {} out of bounds for tensor of dimension {}", axis, shape.len()));
                    }
                    if shape[axis] != 1 {
                        return Err(format!("Cannot squeeze axis {} of size {}, size must be 1", axis, shape[axis]));
                    }
                }
                axes
            }
            None => {
                // Find all dimensions with size 1
                shape.iter().enumerate()
                    .filter_map(|(i, &size)| if size == 1 { Some(i) } else { None })
                    .collect()
            }
        };

        // Create new shape and strides without squeezed dimensions
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        
        for (i, (&size, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
            if !axes_to_squeeze.contains(&i) {
                new_shape.push(size);
                new_strides.push(stride);
            }
        }

        // Handle scalar case
        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }
        
        let new_meta = WasmTensorMeta::new(
            self.meta.dtype(),
            new_shape,
            new_strides,
            self.meta.size(),
            self.meta.offset(),
        );

        Ok(WasmViewDescriptor {
            buffer: Arc::clone(&self.buffer),
            buffer_id: self.buffer_id,
            view_type: ViewType::Squeeze,
            meta: new_meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: Some(Arc::new(self.clone())),
        })
    }

    /// Create an unsqueeze view (add dimension of size 1)
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, String> {
        let shape = self.meta.shape();
        let strides = self.meta.strides();
        
        let new_ndim = shape.len() + 1;
        if axis > shape.len() {
            return Err(format!("Axis {} out of bounds for output tensor of dimension {}", axis, new_ndim));
        }

        // Insert new dimension
        let mut new_shape = shape.clone();
        new_shape.insert(axis, 1);
        
        // Calculate stride for new dimension
        let mut new_strides = strides.clone();
        let new_stride = if axis < strides.len() {
            strides[axis] * shape[axis]
        } else {
            1
        };
        new_strides.insert(axis, new_stride);
        
        let new_meta = WasmTensorMeta::new(
            self.meta.dtype(),
            new_shape,
            new_strides,
            self.meta.size(),
            self.meta.offset(),
        );

        Ok(WasmViewDescriptor {
            buffer: Arc::clone(&self.buffer),
            buffer_id: self.buffer_id,
            view_type: ViewType::Unsqueeze,
            meta: new_meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: Some(Arc::new(self.clone())),
        })
    }

    /// Create an expand view (broadcast dimensions)
    pub fn expand(&self, new_shape: Vec<i32>) -> Result<Self, String> {
        let shape = self.meta.shape();
        let strides = self.meta.strides();
        
        // Handle dimension mismatch by prepending 1s
        let mut expanded_shape = shape.clone();
        let mut expanded_strides = strides.clone();
        
        if new_shape.len() > shape.len() {
            let diff = new_shape.len() - shape.len();
            for _ in 0..diff {
                expanded_shape.insert(0, 1);
                expanded_strides.insert(0, 0);
            }
        }

        // Process expansion
        let mut final_shape = Vec::new();
        let mut final_strides = Vec::new();
        
        for (i, &target_size) in new_shape.iter().enumerate() {
            let current_size = expanded_shape[i];
            let current_stride = expanded_strides[i];
            
            if target_size == -1 {
                // Keep current dimension
                final_shape.push(current_size);
                final_strides.push(current_stride);
            } else if target_size as usize == current_size {
                // Dimension already correct size
                final_shape.push(current_size);
                final_strides.push(current_stride);
            } else if current_size == 1 {
                // Broadcast dimension
                final_shape.push(target_size as usize);
                final_strides.push(0); // Zero stride for broadcasting
            } else {
                return Err(format!(
                    "Cannot expand dimension {} from size {} to {}",
                    i, current_size, target_size
                ));
            }
        }

        // Calculate new total size
        let new_size = final_shape.iter().product();
        
        let new_meta = WasmTensorMeta::new(
            self.meta.dtype(),
            final_shape,
            final_strides,
            new_size,
            self.meta.offset(),
        );

        Ok(WasmViewDescriptor {
            buffer: Arc::clone(&self.buffer),
            buffer_id: self.buffer_id,
            view_type: ViewType::Expand,
            meta: new_meta,
            ref_count: Arc::new(AtomicUsize::new(1)),
            parent: Some(Arc::new(self.clone())),
        })
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let shape = self.meta.shape();
        let strides = self.meta.strides();
        
        if shape.is_empty() {
            return true;
        }

        // Check C-contiguous (row-major)
        let mut expected_stride = 1;
        for i in (0..shape.len()).rev() {
            if strides[i] != expected_stride {
                return false;
            }
            expected_stride *= shape[i];
        }
        
        true
    }

    /// Get a slice of the data (for when zero-copy isn't possible)
    pub fn requires_copy(&self) -> bool {
        match self.view_type {
            ViewType::Base | ViewType::Reshape | ViewType::Squeeze | ViewType::Unsqueeze => false,
            ViewType::Transpose | ViewType::Permute => !self.is_contiguous(),
            ViewType::Slice => true, // For now, slices require copying
            ViewType::Expand => false, // Broadcasting doesn't require copy
        }
    }

    /// Increment reference count
    pub fn inc_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement reference count
    pub fn dec_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Relaxed)
    }
}

/// Calculate strides for row-major (C-contiguous) layout
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_strides() {
        assert_eq!(calculate_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(calculate_strides(&[5, 7]), vec![7, 1]);
        assert_eq!(calculate_strides(&[10]), vec![1]);
    }

    #[test]
    fn test_reshape_valid() {
        let mut manager = WasmMemoryManager::new();
        let buffer = manager.create_buffer_with_data(&[0u8; 96]).unwrap();
        let meta = WasmTensorMeta::new(
            WasmDType::Float32,
            vec![2, 3, 4],
            vec![12, 4, 1],
            24,
            0,
        );
        
        let view = WasmViewDescriptor::new_base(buffer, meta);
        
        // Valid reshapes
        assert!(view.reshape(vec![6, 4]).is_ok());
        assert!(view.reshape(vec![24]).is_ok());
        assert!(view.reshape(vec![4, 6]).is_ok());
        
        // Invalid reshapes
        assert!(view.reshape(vec![5, 5]).is_err());
        assert!(view.reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn test_transpose() {
        let mut manager = WasmMemoryManager::new();
        let buffer = manager.create_buffer_with_data(&[0u8; 24]).unwrap();
        let meta = WasmTensorMeta::new(
            WasmDType::Float32,
            vec![2, 3],
            vec![3, 1],
            6,
            0,
        );
        
        let view = WasmViewDescriptor::new_base(buffer, meta);
        let transposed = view.transpose().unwrap();
        
        assert_eq!(transposed.meta.shape(), vec![3, 2]);
        assert_eq!(transposed.meta.strides(), vec![1, 3]);
    }

    #[test]
    fn test_squeeze() {
        let mut manager = WasmMemoryManager::new();
        let buffer1 = manager.create_buffer_with_data(&[0u8; 24]).unwrap();
        let meta1 = WasmTensorMeta::new(
            WasmDType::Float32,
            vec![1, 3, 1, 4, 1],
            vec![12, 4, 4, 1, 1],
            12,
            0,
        );
        
        let view = WasmViewDescriptor::new_base(buffer1, meta1);
        
        // Squeeze all
        let squeezed = view.squeeze(None).unwrap();
        assert_eq!(squeezed.meta.shape(), vec![3, 4]);
        
        // Squeeze specific
        let buffer2 = manager.create_buffer_with_data(&[0u8; 24]).unwrap();
        let meta2 = WasmTensorMeta::new(
            WasmDType::Float32,
            vec![1, 3, 1, 4, 1],
            vec![12, 4, 4, 1, 1],
            12,
            0,
        );
        let view2 = WasmViewDescriptor::new_base(buffer2, meta2);
        let squeezed2 = view2.squeeze(Some(vec![0, 2])).unwrap();
        assert_eq!(squeezed2.meta.shape(), vec![3, 4, 1]);
    }

    #[test]
    fn test_expand() {
        let mut manager = WasmMemoryManager::new();
        let buffer = manager.create_buffer_with_data(&[0u8; 12]).unwrap();
        let meta = WasmTensorMeta::new(
            WasmDType::Float32,
            vec![3, 1],
            vec![1, 1],
            3,
            0,
        );
        
        let view = WasmViewDescriptor::new_base(buffer, meta);
        let expanded = view.expand(vec![3, 4]).unwrap();
        
        assert_eq!(expanded.meta.shape(), vec![3, 4]);
        assert_eq!(expanded.meta.strides(), vec![1, 0]); // 0 stride for broadcast
    }
}