/*!
 * Operation pattern recognition and optimization for arena allocation
 * 
 * Implements ONNX Runtime-style memory pattern optimization:
 * - Hash operation sequences + input shapes
 * - Cache allocation requirements for repeated patterns  
 * - Enable bulk pre-allocation for 1.7-4.5x speedup
 */

use std::collections::HashMap;
use std::hash::{Hash, Hasher, DefaultHasher};
use wasm_bindgen::prelude::*;
use crate::types::{WasmOperation, WasmDType};

/// Unique identifier for an operation pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatternId(u64);

impl PatternId {
    /// Create new pattern ID (for testing)
    pub fn new(id: u64) -> Self {
        PatternId(id)
    }
}

/// Description of a single operation in a sequence
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OperationDesc {
    pub operation: WasmOperation,
    pub input_shapes: Vec<Vec<usize>>,
    pub input_dtypes: Vec<WasmDType>,
    pub output_shape: Vec<usize>,
    pub output_dtype: WasmDType,
}

/// Memory allocation requirement for a single tensor
#[derive(Debug, Clone)]
pub struct AllocationRequirement {
    pub size_bytes: usize,
    pub alignment: usize,
    pub is_output: bool,  // true if this is an operation output, false if intermediate
}

/// Cached pattern containing allocation strategy
#[derive(Debug, Clone)]
pub struct OperationPattern {
    pub pattern_id: PatternId,
    pub operations: Vec<OperationDesc>,
    pub allocations: Vec<AllocationRequirement>,
    pub total_memory_needed: usize,
    pub estimated_speedup: f32,  // Based on hit count and complexity
}

/// Pattern signature for rapid lookup and matching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PatternSignature {
    pub operation: WasmOperation,
    pub input_shapes: Vec<Vec<usize>>,
    pub input_dtypes: Vec<WasmDType>,
}

impl PatternSignature {
    pub fn new(operation: WasmOperation, input_shapes: Vec<Vec<usize>>, input_dtypes: Vec<WasmDType>) -> Self {
        PatternSignature {
            operation,
            input_shapes,
            input_dtypes,
        }
    }
    
    /// Create signature from operation description
    pub fn from_operation_desc(desc: &OperationDesc) -> Self {
        PatternSignature {
            operation: desc.operation,
            input_shapes: desc.input_shapes.clone(),
            input_dtypes: desc.input_dtypes.clone(),
        }
    }
}

/// Pattern cache with LRU eviction and hit counting
pub struct PatternCache {
    patterns: HashMap<PatternId, OperationPattern>,
    hit_counts: HashMap<PatternId, usize>,
    last_used: HashMap<PatternId, u64>,  // Timestamp for LRU
    current_time: u64,
    max_patterns: usize,
    max_memory_usage: usize,
    current_memory_usage: usize,
}

impl PatternCache {
    /// Create new pattern cache
    pub fn new(max_patterns: usize, max_memory_mb: usize) -> Self {
        PatternCache {
            patterns: HashMap::new(),
            hit_counts: HashMap::new(),
            last_used: HashMap::new(),
            current_time: 0,
            max_patterns,
            max_memory_usage: max_memory_mb * 1024 * 1024,
            current_memory_usage: 0,
        }
    }
    
    /// Hash a sequence of operations to create pattern ID
    pub fn hash_operation_sequence(&self, operations: &[OperationDesc]) -> PatternId {
        let mut hasher = DefaultHasher::new();
        
        // Hash the operation sequence
        operations.len().hash(&mut hasher);
        for op in operations {
            op.hash(&mut hasher);
        }
        
        PatternId(hasher.finish())
    }
    
    /// Check if pattern exists in cache (and update access time)
    pub fn get_pattern(&mut self, pattern_id: PatternId) -> Option<&OperationPattern> {
        if let Some(pattern) = self.patterns.get(&pattern_id) {
            // Update access tracking
            self.current_time += 1;
            self.last_used.insert(pattern_id, self.current_time);
            *self.hit_counts.entry(pattern_id).or_insert(0) += 1;
            
            Some(pattern)
        } else {
            None
        }
    }
    
    /// Store new pattern in cache
    pub fn store_pattern(&mut self, pattern: OperationPattern) -> Result<(), String> {
        let pattern_id = pattern.pattern_id;
        let pattern_memory = pattern.total_memory_needed;
        
        // Check memory limits
        if self.current_memory_usage + pattern_memory > self.max_memory_usage {
            self.evict_patterns(pattern_memory)?;
        }
        
        // Check pattern count limits
        if self.patterns.len() >= self.max_patterns {
            self.evict_lru_pattern()?;
        }
        
        // Store pattern
        self.current_memory_usage += pattern_memory;
        self.current_time += 1;
        self.last_used.insert(pattern_id, self.current_time);
        self.hit_counts.insert(pattern_id, 0);
        self.patterns.insert(pattern_id, pattern);
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> PatternCacheStats {
        let total_hits: usize = self.hit_counts.values().sum();
        let hot_patterns = self.hit_counts.iter()
            .filter(|(_, &count)| count > 1)
            .count();
            
        PatternCacheStats {
            pattern_count: self.patterns.len(),
            total_hits,
            hot_patterns,
            memory_usage_bytes: self.current_memory_usage,
            memory_utilization: self.current_memory_usage as f32 / self.max_memory_usage as f32,
        }
    }
    
    /// Clear all cached patterns
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.hit_counts.clear();
        self.last_used.clear();
        self.current_memory_usage = 0;
        self.current_time = 0;
    }
    
    /// Find pattern starting with given operation signature
    pub fn find_matching_pattern(&mut self, signature: &PatternSignature) -> Option<PatternId> {
        // Look for patterns that start with the given signature
        for (pattern_id, pattern) in &self.patterns {
            if let Some(first_op) = pattern.operations.first() {
                let first_signature = PatternSignature::from_operation_desc(first_op);
                if first_signature == *signature {
                    // Update access tracking
                    self.current_time += 1;
                    self.last_used.insert(*pattern_id, self.current_time);
                    *self.hit_counts.entry(*pattern_id).or_insert(0) += 1;
                    return Some(*pattern_id);
                }
            }
        }
        None
    }
    
    /// Update pattern with execution results
    pub fn update_pattern_stats(&mut self, pattern_id: PatternId, execution_time: f32) {
        if let Some(pattern) = self.patterns.get_mut(&pattern_id) {
            // Update estimated speedup based on actual execution time
            // This is a simple moving average - in practice you'd want more sophisticated tracking
            let current_speedup = pattern.estimated_speedup;
            let base_execution_time = execution_time / current_speedup; // Estimate base time
            let new_speedup = if execution_time > 0.0 {
                base_execution_time / execution_time
            } else {
                current_speedup
            };
            
            // Use exponential moving average to update speedup estimate
            pattern.estimated_speedup = current_speedup * 0.9 + new_speedup * 0.1;
        }
    }
    
    /// Hash a pattern signature to create pattern ID
    pub fn hash_signature(&self, signature: &PatternSignature) -> PatternId {
        let mut hasher = DefaultHasher::new();
        signature.hash(&mut hasher);
        PatternId(hasher.finish())
    }
    
    /// Evict least recently used pattern
    fn evict_lru_pattern(&mut self) -> Result<(), String> {
        if self.patterns.is_empty() {
            return Err("No patterns to evict".to_string());
        }
        
        // Find LRU pattern
        let lru_pattern_id = self.last_used.iter()
            .min_by_key(|(_, &time)| time)
            .map(|(id, _)| *id)
            .ok_or("No patterns to evict")?;
        
        // Remove pattern and update tracking
        if let Some(pattern) = self.patterns.remove(&lru_pattern_id) {
            self.current_memory_usage = self.current_memory_usage
                .saturating_sub(pattern.total_memory_needed);
        }
        self.hit_counts.remove(&lru_pattern_id);
        self.last_used.remove(&lru_pattern_id);
        
        Ok(())
    }
    
    /// Evict patterns until we have enough space
    fn evict_patterns(&mut self, needed_space: usize) -> Result<(), String> {
        let target_usage = self.max_memory_usage.saturating_sub(needed_space);
        
        // Sort patterns by access time (LRU first)
        let mut patterns_by_time: Vec<_> = self.last_used.iter()
            .map(|(id, time)| (*id, *time))
            .collect();
        patterns_by_time.sort_by_key(|(_, time)| *time);
        
        // Evict patterns until we have enough space
        for (pattern_id, _) in patterns_by_time {
            if self.current_memory_usage <= target_usage {
                break;
            }
            
            if let Some(pattern) = self.patterns.remove(&pattern_id) {
                self.current_memory_usage = self.current_memory_usage
                    .saturating_sub(pattern.total_memory_needed);
            }
            self.hit_counts.remove(&pattern_id);
            self.last_used.remove(&pattern_id);
        }
        
        if self.current_memory_usage + needed_space > self.max_memory_usage {
            return Err("Cannot free enough space for new pattern".to_string());
        }
        
        Ok(())
    }
}

/// Cache performance statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct PatternCacheStats {
    pattern_count: usize,
    total_hits: usize,
    hot_patterns: usize,  // Patterns with >1 hit
    memory_usage_bytes: usize,
    memory_utilization: f32,
}

#[wasm_bindgen]
impl PatternCacheStats {
    #[wasm_bindgen(getter)]
    pub fn pattern_count(&self) -> usize { self.pattern_count }
    
    #[wasm_bindgen(getter)]
    pub fn total_hits(&self) -> usize { self.total_hits }
    
    #[wasm_bindgen(getter)]
    pub fn hot_patterns(&self) -> usize { self.hot_patterns }
    
    #[wasm_bindgen(getter)]
    pub fn memory_usage_bytes(&self) -> usize { self.memory_usage_bytes }
    
    #[wasm_bindgen(getter)]
    pub fn memory_utilization(&self) -> f32 { self.memory_utilization }
}

/// Pattern builder for creating operation patterns
pub struct PatternBuilder {
    operations: Vec<OperationDesc>,
    allocations: Vec<AllocationRequirement>,
    total_memory: usize,
}

impl PatternBuilder {
    pub fn new() -> Self {
        PatternBuilder {
            operations: Vec::new(),
            allocations: Vec::new(),
            total_memory: 0,
        }
    }
    
    /// Add operation to pattern
    pub fn add_operation(&mut self, operation: OperationDesc) -> &mut Self {
        self.operations.push(operation);
        self
    }
    
    /// Add allocation requirement
    pub fn add_allocation(&mut self, allocation: AllocationRequirement) -> &mut Self {
        self.total_memory += allocation.size_bytes;
        self.allocations.push(allocation);
        self
    }
    
    /// Build the final pattern
    pub fn build(self, cache: &PatternCache) -> OperationPattern {
        let pattern_id = cache.hash_operation_sequence(&self.operations);
        
        // Estimate speedup based on operation complexity
        let estimated_speedup = calculate_estimated_speedup(&self.operations);
        
        OperationPattern {
            pattern_id,
            operations: self.operations,
            allocations: self.allocations,
            total_memory_needed: self.total_memory,
            estimated_speedup,
        }
    }
}

/// Calculate estimated speedup for a pattern based on complexity
fn calculate_estimated_speedup(operations: &[OperationDesc]) -> f32 {
    let base_speedup = 1.5; // Base speedup from avoiding individual allocations
    let complexity_factor = operations.len() as f32 * 0.1; // More ops = more benefit
    let memory_factor = operations.iter()
        .map(|op| op.input_shapes.iter().map(|s| s.iter().product::<usize>()).sum::<usize>())
        .sum::<usize>() as f32 / 1000.0; // Larger tensors = more benefit
    
    (base_speedup + complexity_factor + (memory_factor * 0.001)).min(4.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_cache_basic() {
        let mut cache = PatternCache::new(10, 100); // 10 patterns, 100MB
        
        // Create a simple pattern
        let mut builder = PatternBuilder::new();
        builder.add_operation(OperationDesc {
            operation: WasmOperation::Add,
            input_shapes: vec![vec![10, 20], vec![10, 20]],
            input_dtypes: vec![WasmDType::Float32, WasmDType::Float32],
            output_shape: vec![10, 20],
            output_dtype: WasmDType::Float32,
        });
        builder.add_allocation(AllocationRequirement {
            size_bytes: 800,
            alignment: 16,
            is_output: true,
        });
        
        let pattern = builder.build(&cache);
        let pattern_id = pattern.pattern_id;
        
        // Store pattern
        cache.store_pattern(pattern).unwrap();
        
        // Retrieve pattern (should increment hit count)
        let retrieved = cache.get_pattern(pattern_id).unwrap();
        assert_eq!(retrieved.pattern_id, pattern_id);
        
        // Check stats
        let stats = cache.cache_stats();
        assert_eq!(stats.pattern_count, 1);
        assert_eq!(stats.total_hits, 1);
    }
    
    #[test]
    fn test_pattern_hashing() {
        let cache = PatternCache::new(10, 100);
        
        let op1 = OperationDesc {
            operation: WasmOperation::Add,
            input_shapes: vec![vec![10, 20]],
            input_dtypes: vec![WasmDType::Float32],
            output_shape: vec![10, 20],
            output_dtype: WasmDType::Float32,
        };
        
        let op2 = OperationDesc {
            operation: WasmOperation::Mul,
            input_shapes: vec![vec![10, 20]],
            input_dtypes: vec![WasmDType::Float32],
            output_shape: vec![10, 20],
            output_dtype: WasmDType::Float32,
        };
        
        let hash1 = cache.hash_operation_sequence(&[op1.clone()]);
        let hash2 = cache.hash_operation_sequence(&[op2]);
        let hash3 = cache.hash_operation_sequence(&[op1]);
        
        // Same operation should produce same hash
        assert_eq!(hash1, hash3);
        // Different operations should produce different hashes
        assert_ne!(hash1, hash2);
    }
    
    #[test]
    fn test_lru_eviction() {
        let mut cache = PatternCache::new(2, 100); // Only 2 patterns max
        
        // Create 3 patterns
        let patterns: Vec<_> = (0..3).map(|i| {
            let mut builder = PatternBuilder::new();
            builder.add_operation(OperationDesc {
                operation: WasmOperation::Add,
                input_shapes: vec![vec![i + 1, 10]], // Different shapes for different hashes
                input_dtypes: vec![WasmDType::Float32],
                output_shape: vec![i + 1, 10],
                output_dtype: WasmDType::Float32,
            });
            builder.add_allocation(AllocationRequirement {
                size_bytes: 100,
                alignment: 16,
                is_output: true,
            });
            builder.build(&cache)
        }).collect();
        
        // Store first 2 patterns
        cache.store_pattern(patterns[0].clone()).unwrap();
        cache.store_pattern(patterns[1].clone()).unwrap();
        assert_eq!(cache.patterns.len(), 2);
        
        // Access first pattern to make it more recently used
        cache.get_pattern(patterns[0].pattern_id);
        
        // Store third pattern - should evict the second one (LRU)
        cache.store_pattern(patterns[2].clone()).unwrap();
        assert_eq!(cache.patterns.len(), 2);
        
        // First and third should exist, second should be evicted
        assert!(cache.patterns.contains_key(&patterns[0].pattern_id));
        assert!(!cache.patterns.contains_key(&patterns[1].pattern_id));
        assert!(cache.patterns.contains_key(&patterns[2].pattern_id));
    }
    
    // WASM-specific integration tests
    #[cfg(test)]
    mod wasm_tests {
        use super::*;
        use wasm_bindgen_test::*;
        
        #[wasm_bindgen_test]
        fn wasm_test_pattern_cache_creation() {
            let cache = PatternCache::new(10, 100); // 10 patterns, 100MB
            let stats = cache.cache_stats();
            assert_eq!(stats.pattern_count(), 0);
            assert_eq!(stats.total_hits(), 0);
            assert_eq!(stats.hot_patterns(), 0);
            assert_eq!(stats.memory_usage_bytes(), 0);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_pattern_signature_matching() {
            let mut cache = PatternCache::new(5, 50);
            
            // Create a signature for a binary add operation
            let signature = PatternSignature::new(
                WasmOperation::Add,
                vec![vec![10, 20], vec![10, 20]], // Two [10, 20] inputs
                vec![WasmDType::Float32, WasmDType::Float32],
            );
            
            // Should not find any pattern initially
            assert!(cache.find_matching_pattern(&signature).is_none());
            
            // Create and store a pattern that starts with this signature
            let mut builder = PatternBuilder::new();
            builder.add_operation(OperationDesc {
                operation: WasmOperation::Add,
                input_shapes: vec![vec![10, 20], vec![10, 20]],
                input_dtypes: vec![WasmDType::Float32, WasmDType::Float32],
                output_shape: vec![10, 20],
                output_dtype: WasmDType::Float32,
            });
            builder.add_allocation(AllocationRequirement {
                size_bytes: 800, // 10 * 20 * 4 bytes
                alignment: 16,
                is_output: true,
            });
            
            let pattern = builder.build(&cache);
            cache.store_pattern(pattern).unwrap();
            
            // Now should find the pattern
            let found_pattern_id = cache.find_matching_pattern(&signature);
            assert!(found_pattern_id.is_some());
            
            // Check that accessing the pattern increases hit count
            let stats = cache.cache_stats();
            assert_eq!(stats.pattern_count(), 1);
            assert_eq!(stats.total_hits(), 1);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_pattern_speedup_estimation() {
            let cache = PatternCache::new(10, 100);
            
            // Create patterns with different complexity levels
            let simple_ops = vec![OperationDesc {
                operation: WasmOperation::Add,
                input_shapes: vec![vec![10]],
                input_dtypes: vec![WasmDType::Float32],
                output_shape: vec![10],
                output_dtype: WasmDType::Float32,
            }];
            
            let complex_ops = vec![
                OperationDesc {
                    operation: WasmOperation::Matmul,
                    input_shapes: vec![vec![100, 100], vec![100, 100]],
                    input_dtypes: vec![WasmDType::Float32, WasmDType::Float32],
                    output_shape: vec![100, 100],
                    output_dtype: WasmDType::Float32,
                },
                OperationDesc {
                    operation: WasmOperation::Add,
                    input_shapes: vec![vec![100, 100], vec![100, 100]],
                    input_dtypes: vec![WasmDType::Float32, WasmDType::Float32],
                    output_shape: vec![100, 100],
                    output_dtype: WasmDType::Float32,
                },
            ];
            
            let simple_speedup = calculate_estimated_speedup(&simple_ops);
            let complex_speedup = calculate_estimated_speedup(&complex_ops);
            
            // Complex operations should have higher estimated speedup
            assert!(complex_speedup > simple_speedup);
            assert!(simple_speedup >= 1.5); // Base speedup
            assert!(complex_speedup <= 4.5); // Max speedup cap
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_pattern_cache_memory_management() {
            // Create cache with very small memory limit to force eviction
            let mut cache = PatternCache::new(10, 0); // 0MB memory limit - should reject all patterns
            
            let mut builder = PatternBuilder::new();
            builder.add_operation(OperationDesc {
                operation: WasmOperation::Add,
                input_shapes: vec![vec![10, 10]], 
                input_dtypes: vec![WasmDType::Float32],
                output_shape: vec![10, 10],
                output_dtype: WasmDType::Float32,
            });
            
            // Add any allocation requirement - even small ones should fail with 0MB limit
            builder.add_allocation(AllocationRequirement {
                size_bytes: 1024, // 1KB should exceed 0MB limit
                alignment: 16,
                is_output: true,
            });
            
            let pattern = builder.build(&cache);
            
            // Should fail to store due to memory limit
            let result = cache.store_pattern(pattern);
            assert!(result.is_err());
            
            let stats = cache.cache_stats();
            assert_eq!(stats.pattern_count(), 0);
            assert_eq!(stats.memory_usage_bytes(), 0);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_pattern_update_stats() {
            let mut cache = PatternCache::new(10, 100);
            
            // Create and store a pattern
            let mut builder = PatternBuilder::new();
            builder.add_operation(OperationDesc {
                operation: WasmOperation::Mul,
                input_shapes: vec![vec![5, 5], vec![5, 5]],
                input_dtypes: vec![WasmDType::Float32, WasmDType::Float32],
                output_shape: vec![5, 5],
                output_dtype: WasmDType::Float32,
            });
            builder.add_allocation(AllocationRequirement {
                size_bytes: 100,
                alignment: 16,
                is_output: true,
            });
            
            let pattern = builder.build(&cache);
            let pattern_id = pattern.pattern_id;
            let initial_speedup = pattern.estimated_speedup;
            
            cache.store_pattern(pattern).unwrap();
            
            // Update pattern stats with execution time
            cache.update_pattern_stats(pattern_id, 10.0); // 10ms execution time
            
            // Get updated pattern and check speedup was updated
            let updated_pattern = cache.get_pattern(pattern_id).unwrap();
            
            // Speedup should be different (exponential moving average)
            assert!(updated_pattern.estimated_speedup != initial_speedup);
        }
    }
}