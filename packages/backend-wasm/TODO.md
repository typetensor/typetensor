# WASM Backend Memory Management: Issues and Fixes

## Overview

This document outlines critical memory management issues discovered in the TypeTensor WASM backend, organized by severity and with a Test-Driven Development (TDD) approach for fixes.

## Critical Issues (Must Fix)

### 1. View Lifetime Management - Use After Free

**Severity**: 🔴 CRITICAL - Memory Safety

**Finding**: Zero-copy views have no connection to buffer lifetime, allowing access to freed memory.

**Key Files**:
- `src/memory-views.ts`: Lines 40-79 (createView method)
- `src/device.ts`: Lines 306-328 (readDataView method)
- `src/data.ts`: Lines 53-81 (dispose method)

**Problem Code**:
```typescript
// src/memory-views.ts:40-79
createView(ptr: number, size: number, dtype: DType<any, any, any>): ArrayBufferView {
  // Creates view without any reference to the buffer
  // No way to know if buffer is still alive
  switch (dtype) {
    case float32:
      return new Float32Array(buffer, byteOffset, size);
    // ...
  }
}

// src/device.ts:306-328
readDataView(data: DeviceData, dtype: DType<any, any, any>): ArrayBufferView {
  // Returns view that outlives the buffer
  return this.memoryViewManager!.createView(ptr, size / dtype.__byteSize, dtype);
}
```

**Test to Demonstrate Issue**:
```typescript
// test/memory-safety.test.ts
describe('View Lifetime Management', () => {
  it('should prevent use-after-free when buffer is disposed', () => {
    const device = await WASMDevice.create();
    const data = device.createData(1024);
    const view = device.readDataView(data, float32);
    
    // This should work
    view[0] = 42;
    expect(view[0]).toBe(42);
    
    // Dispose the buffer
    device.disposeData(data);
    
    // This should throw or return undefined, NOT access freed memory
    expect(() => view[0] = 123).toThrow('View is no longer valid: buffer has been disposed');
    expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
  });

  it('should invalidate all views when buffer is released to pool', () => {
    const device = await WASMDevice.create();
    const data1 = device.createData(1024);
    const view1 = device.readDataView(data1, float32);
    
    device.disposeData(data1);
    
    // Create new buffer - might reuse same memory!
    const data2 = device.createData(1024);
    const view2 = device.readDataView(data2, float32);
    
    // view1 should not see data2's values
    view2[0] = 999;
    expect(() => view1[0]).toThrow();
  });
});
```

**Suggested Fix**:

1. Add buffer lifecycle tracking to views:
```typescript
// src/memory-views.ts
interface TrackedView {
  view: ArrayBufferView;
  bufferId: string;
  generation: number;
}

class MemoryViewManager {
  private activeViews = new Map<string, Set<WeakRef<TrackedView>>>();
  private bufferGenerations = new Map<string, number>();

  createView(bufferId: string, ptr: number, size: number, dtype: DType): ArrayBufferView {
    const generation = this.bufferGenerations.get(bufferId) || 0;
    const rawView = this.createRawView(ptr, size, dtype);
    
    // Wrap in a Proxy to check validity on access
    const trackedView: TrackedView = {
      view: rawView,
      bufferId,
      generation
    };
    
    // Track this view
    if (!this.activeViews.has(bufferId)) {
      this.activeViews.set(bufferId, new Set());
    }
    this.activeViews.get(bufferId)!.add(new WeakRef(trackedView));
    
    return new Proxy(rawView, {
      get: (target, prop) => {
        if (!this.isViewValid(trackedView)) {
          throw new Error('View is no longer valid: buffer has been disposed');
        }
        return target[prop];
      },
      set: (target, prop, value) => {
        if (!this.isViewValid(trackedView)) {
          throw new Error('View is no longer valid: buffer has been disposed');
        }
        target[prop] = value;
        return true;
      }
    });
  }

  invalidateBuffer(bufferId: string): void {
    // Increment generation to invalidate all views
    const currentGen = this.bufferGenerations.get(bufferId) || 0;
    this.bufferGenerations.set(bufferId, currentGen + 1);
  }

  private isViewValid(trackedView: TrackedView): boolean {
    const currentGen = this.bufferGenerations.get(trackedView.bufferId) || 0;
    return trackedView.generation === currentGen && 
           trackedView.view.buffer === this.memory.buffer;
  }
}
```

2. Update device.ts to pass buffer ID:
```typescript
// src/device.ts
readDataView(data: DeviceData, dtype: DType<any, any, any>): ArrayBufferView {
  const wasmData = data as WASMDeviceData;
  const bufferId = wasmData.id;
  // ... get ptr and size ...
  return this.memoryViewManager!.createView(bufferId, ptr, size / dtype.__byteSize, dtype);
}

disposeData(data: DeviceData): void {
  const wasmData = data as WASMDeviceData;
  // Invalidate all views before releasing
  this.memoryViewManager!.invalidateBuffer(wasmData.id);
  // ... rest of disposal ...
}
```

**Test to Verify Fix**:
```typescript
describe('Fixed View Lifetime Management', () => {
  it('should properly track view validity', () => {
    const device = await WASMDevice.create();
    const data = device.createData(1024);
    const view = device.readDataView(data, float32);
    
    view[0] = 42;
    expect(view[0]).toBe(42);
    
    device.disposeData(data);
    
    // Now properly throws
    expect(() => view[0] = 123).toThrow('View is no longer valid: buffer has been disposed');
  });
});
```

---

### 2. Incomplete Zero-Copy Operations

**Severity**: 🔴 CRITICAL - Silent Failure

**Finding**: Core tensor operations (reshape, view, flatten, etc.) return success without implementation.

**Key Files**:
- `wasm/src/operations/view.rs`: Lines 21-44
- `wasm/src/operations/mod.rs`: Lines 143-156

**Problem Code**:
```rust
// wasm/src/operations/view.rs:21-44
match operation {
    WasmOperation::Reshape | WasmOperation::View | WasmOperation::Flatten => {
        // These are zero-copy operations - output buffer shares data with input
        // The shape change is handled at the metadata level
        // TODO: Implement proper zero-copy views
        Ok(())  // RETURNS SUCCESS WITHOUT DOING ANYTHING!
    }
    WasmOperation::Squeeze | WasmOperation::Unsqueeze => {
        // Zero-copy operations - dimension addition/removal
        // TODO: Implement proper zero-copy views
        Ok(())  // RETURNS SUCCESS WITHOUT DOING ANYTHING!
    }
    // ...
}
```

**Test to Demonstrate Issue**:
```typescript
// test/view-operations.test.ts
describe('View Operations', () => {
  it('should properly reshape tensors', async () => {
    const device = await WASMDevice.create();
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const tensor = tt.tensor(data).on(device);
    
    const reshaped = tensor.reshape([2, 3]);
    const result = await reshaped.data();
    
    // This test will fail - reshape doesn't actually work!
    expect(result.shape).toEqual([2, 3]);
    expect(result.data).toEqual(data);
    
    // Verify it's a view, not a copy
    reshaped.data[0] = 999;
    const originalData = await tensor.data();
    expect(originalData[0]).toBe(999); // Should see the change
  });

  it('should properly flatten tensors', async () => {
    const device = await WASMDevice.create();
    const tensor = tt.tensor([[1, 2, 3], [4, 5, 6]]).on(device);
    
    const flattened = tensor.flatten();
    const result = await flattened.data();
    
    expect(result.shape).toEqual([6]);
    expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
```

**Suggested Fix**:

1. Implement proper view operations in Rust:
```rust
// wasm/src/operations/view.rs
pub fn execute_view_op(
    memory_manager: &mut WasmMemoryManager,
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
) -> WasmResult<()> {
    match operation {
        WasmOperation::Reshape | WasmOperation::View | WasmOperation::Flatten => {
            // For zero-copy views, we need to share the same buffer
            // but with different metadata
            execute_zero_copy_view(
                memory_manager,
                input,
                output,
                input_meta,
                output_meta
            )
        }
        // ...
    }
}

fn execute_zero_copy_view(
    memory_manager: &mut WasmMemoryManager,
    input: &WasmBufferHandle,
    output: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output_meta: &WasmTensorMeta,
) -> WasmResult<()> {
    // Verify the view is valid (same number of elements)
    if input_meta.size() != output_meta.size() {
        return Err(WasmError::InvalidShape);
    }
    
    // For true zero-copy, output should reference the same memory
    // This requires updating the architecture to support shared buffers
    // For now, we'll copy but mark this as a TODO
    
    let input_ptr = memory_manager.get_read_ptr(input);
    let output_ptr = memory_manager.get_write_ptr(output);
    let byte_size = input_meta.byte_size();
    
    unsafe {
        std::ptr::copy_nonoverlapping(input_ptr, output_ptr, byte_size);
    }
    
    // TODO: Implement true zero-copy by sharing buffer handles
    // This requires:
    // 1. Reference counting at the buffer level
    // 2. Metadata-only operations
    // 3. Stride support for non-contiguous views
    
    Ok(())
}
```

2. Add proper view support to TypeScript:
```typescript
// src/device.ts
// Add support for view operations that share buffers
interface ViewDescriptor {
  sourceBuffer: WASMBufferHandle;
  offset: number;
  shape: number[];
  strides: number[];
}

// Update execute method to handle views specially
async execute<T extends AnyStorageTransformation>(
  op: T,
  inputs: DeviceData[],
  output?: DeviceData,
): Promise<DeviceData> {
  if (this.isViewOperation(op.__op)) {
    return this.executeViewOperation(op, inputs[0]!, output);
  }
  // ... existing code ...
}

private isViewOperation(op: string): boolean {
  return ['reshape', 'view', 'flatten', 'squeeze', 'unsqueeze'].includes(op);
}

private executeViewOperation(
  op: AnyStorageTransformation,
  input: DeviceData,
  output?: DeviceData
): DeviceData {
  // For now, create a copy until we have proper view support
  // TODO: Implement zero-copy views
  const inputBuffer = await this.readData(input);
  const outputData = output || this.createData(inputBuffer.byteLength);
  await this.writeData(outputData, inputBuffer);
  return outputData;
}
```

**Test to Verify Fix**:
```typescript
describe('Fixed View Operations', () => {
  it('should reshape without copying data', async () => {
    const device = await WASMDevice.create();
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const tensor = tt.tensor(data).on(device);
    
    const reshaped = tensor.reshape([2, 3]);
    const result = await reshaped.data();
    
    expect(result.shape).toEqual([2, 3]);
    expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
```

---

### 3. Type Safety and Bounds Checking

**Severity**: 🔴 CRITICAL - Memory Corruption Risk

**Finding**: No bounds checking on array accesses, extensive use of `any` type.

**Key Files**:
- `src/device.ts`: Lines 442-443, 514, 523-543
- `src/types.ts`: Lines 78-81

**Problem Code**:
```typescript
// src/device.ts:514
(outputArray as any)[outputFlatIndex] = (inputArray as any)[inputFlatIndex];
// No bounds checking!

// src/types.ts:78-81
export interface WASMModule {
  WasmOperationDispatcher: any;  // No types!
  WasmMemoryManager: any;
  WasmBufferHandle: any;
  WasmTensorMeta: any;
}
```

**Test to Demonstrate Issue**:
```typescript
// test/bounds-safety.test.ts
describe('Bounds Safety', () => {
  it('should handle out-of-bounds access safely', async () => {
    const device = await WASMDevice.create();
    const tensor = tt.tensor([1, 2, 3]).on(device);
    
    // Try to slice with invalid indices
    const badSlice = () => tensor.slice([[0, 10]]); // End > length
    expect(badSlice).toThrow('Slice index 10 is out of bounds for dimension of size 3');
    
    // Try to access with invalid strides
    const badStrides = () => {
      const data = device.createData(12); // 3 float32s
      const view = device.readDataView(data, float32);
      // Manually compute bad index
      const badIndex = 999;
      return view[badIndex]; // Should bounds check!
    };
    expect(badStrides).toThrow('Index 999 out of bounds for array of length 3');
  });
});
```

**Suggested Fix**:

1. Generate proper TypeScript types from WASM:
```typescript
// src/types/wasm-bindings.d.ts
export interface WasmBufferHandle {
  id(): number;
  size(): number;
  ptr(): number;
  is_initialized(): boolean;
  clone_handle(): WasmBufferHandle;
}

export interface WasmTensorMeta {
  constructor(
    dtype: number,
    shape: number[],
    strides: number[],
    size: number,
    offset: number
  ): WasmTensorMeta;
  dtype(): number;
  shape(): number[];
  strides(): number[];
  size(): number;
  offset(): number;
  byte_size(): number;
}

export interface WasmOperationDispatcher {
  create_buffer_with_js_data(data: Uint8Array): WasmBufferHandle;
  release_buffer(handle: WasmBufferHandle): boolean;
  execute_unary(
    op: number,
    input: WasmBufferHandle,
    inputMeta: WasmTensorMeta,
    outputMeta: WasmTensorMeta,
    output: WasmBufferHandle | null
  ): WasmBufferHandle;
  // ... etc
}
```

2. Add bounds checking to array operations:
```typescript
// src/device.ts
private createSlicedBuffer(
  input: DeviceData,
  sliceIndices: SliceIndex[],
  inputShape: readonly number[],
  inputStrides: readonly number[],
  outputShape: readonly number[],
  dtype: any,
): Promise<ArrayBuffer> {
  const inputBuffer = await this.readData(input);
  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const outputBuffer = new ArrayBuffer(outputSize * dtype.__byteSize);

  const inputArray = this.createTypedArray(inputBuffer, dtype);
  const outputArray = this.createTypedArray(outputBuffer, dtype);
  
  // Add bounds checking
  const inputLength = inputArray.length;
  
  for (let outputFlatIndex = 0; outputFlatIndex < outputSize; outputFlatIndex++) {
    const outputIndices = this.flatIndexToIndices(outputFlatIndex, outputShape);
    const inputIndices = this.mapOutputToInputIndices(outputIndices, sliceIndices, inputShape);
    const inputFlatIndex = this.computeFlatIndex(inputIndices, inputStrides);
    
    // BOUNDS CHECK
    if (inputFlatIndex < 0 || inputFlatIndex >= inputLength) {
      throw new Error(`Index ${inputFlatIndex} out of bounds for array of length ${inputLength}`);
    }
    
    outputArray[outputFlatIndex] = inputArray[inputFlatIndex];
  }

  return outputBuffer;
}
```

3. Add validation to slice indices:
```typescript
// src/device.ts
private validateSliceIndices(
  sliceIndices: SliceIndex[],
  shape: readonly number[]
): void {
  for (let i = 0; i < sliceIndices.length; i++) {
    const slice = sliceIndices[i];
    const dimSize = shape[i];
    
    if (typeof slice === 'number') {
      const normalizedIndex = slice < 0 ? dimSize + slice : slice;
      if (normalizedIndex < 0 || normalizedIndex >= dimSize) {
        throw new Error(
          `Slice index ${slice} is out of bounds for dimension of size ${dimSize}`
        );
      }
    } else if (slice && typeof slice === 'object') {
      const start = slice.start ?? 0;
      const stop = slice.stop ?? dimSize;
      const normalizedStart = start < 0 ? dimSize + start : start;
      const normalizedStop = stop < 0 ? dimSize + stop : stop;
      
      if (normalizedStart < 0 || normalizedStart > dimSize) {
        throw new Error(
          `Slice start ${start} is out of bounds for dimension of size ${dimSize}`
        );
      }
      if (normalizedStop < 0 || normalizedStop > dimSize) {
        throw new Error(
          `Slice stop ${stop} is out of bounds for dimension of size ${dimSize}`
        );
      }
    }
  }
}
```

**Test to Verify Fix**:
```typescript
describe('Fixed Bounds Safety', () => {
  it('should validate slice indices', async () => {
    const device = await WASMDevice.create();
    const tensor = tt.tensor([1, 2, 3]).on(device);
    
    expect(() => tensor.slice([[0, 10]])).toThrow('Slice stop 10 is out of bounds');
    expect(() => tensor.slice([[-5, 2]])).toThrow('Slice start -5 is out of bounds');
    
    // Valid slices should work
    const valid = tensor.slice([[0, 2]]);
    expect(await valid.toArray()).toEqual([1, 2]);
  });
});
```

---

## High Priority Issues

### 4. Unnecessary Memory Allocations

**Severity**: 🟡 HIGH - Performance

**Finding**: Multiple unnecessary copies and allocations throughout the codebase.

**Key Files**:
- `src/device.ts`: Lines 226-227, 241, 291-293
- `wasm/src/operations/mod.rs`: Lines 243-248

**Problem Code**:
```typescript
// src/device.ts:226-227
createData(byteLength: number): DeviceData {
  const zeroBuffer = new ArrayBuffer(byteLength);  // Unnecessary!
  const wasmHandle = this.operationDispatcher.create_buffer_with_js_data(new Uint8Array(zeroBuffer));
}

// src/device.ts:241
const sourceBuffer = buffer.slice(0);  // Unnecessary copy!

// src/device.ts:293
return uint8Array.buffer.slice(...);  // Another unnecessary copy!
```

**Test to Demonstrate Issue**:
```typescript
// test/memory-efficiency.test.ts
describe('Memory Efficiency', () => {
  it('should not allocate unnecessary buffers', async () => {
    const device = await WASMDevice.create();
    
    // Mock memory usage tracking
    let allocations = 0;
    const originalArrayBuffer = globalThis.ArrayBuffer;
    globalThis.ArrayBuffer = new Proxy(originalArrayBuffer, {
      construct(target, args) {
        allocations++;
        return Reflect.construct(target, args);
      }
    });
    
    // This should only allocate once in WASM, not in JS
    device.createData(1024);
    expect(allocations).toBe(0); // Currently fails - allocates JS buffer
    
    globalThis.ArrayBuffer = originalArrayBuffer;
  });

  it('should read data without multiple copies', async () => {
    const device = await WASMDevice.create();
    const data = device.createData(1024);
    
    let copies = 0;
    const originalSlice = ArrayBuffer.prototype.slice;
    ArrayBuffer.prototype.slice = function(...args) {
      copies++;
      return originalSlice.apply(this, args);
    };
    
    await device.readData(data);
    expect(copies).toBe(0); // Currently fails - makes unnecessary copies
    
    ArrayBuffer.prototype.slice = originalSlice;
  });
});
```

**Suggested Fix**:

1. Add zero-allocation buffer creation:
```rust
// wasm/src/operations/mod.rs
#[wasm_bindgen]
impl WasmOperationDispatcher {
    /// Create an empty buffer without data copy
    pub fn create_empty_buffer_js(&self, size: usize) -> Result<WasmBufferHandle, JsValue> {
        self.memory_manager
            .borrow_mut()
            .create_empty_buffer(size)
            .map(|(handle, _)| handle)
            .map_err(|e| JsValue::from_str(&e))
    }
}
```

2. Update TypeScript to use zero-allocation:
```typescript
// src/device.ts
createData(byteLength: number): DeviceData {
  this.ensureInitialized();
  
  try {
    // Use WASM-side allocation instead of JS buffer
    const wasmHandle = this.operationDispatcher.create_empty_buffer_js(byteLength);
    return createWASMDeviceData(this, byteLength, wasmHandle);
  } catch (error) {
    throw new Error(`Failed to allocate ${byteLength} bytes on WASM device: ${error}`);
  }
}

createDataWithBuffer(buffer: ArrayBuffer): DeviceData {
  this.ensureInitialized();
  
  try {
    // Don't make defensive copy
    const sourceData = new Uint8Array(buffer);
    const wasmHandle = this.operationDispatcher.create_buffer_with_js_data(sourceData);
    return createWASMDeviceData(this, buffer.byteLength, wasmHandle);
  } catch (error) {
    throw new Error(`Failed to create data with buffer: ${error}`);
  }
}

async readData(data: DeviceData): Promise<ArrayBuffer> {
  // Return the Uint8Array's buffer directly without slice
  const uint8Array = this.operationDispatcher!.copy_buffer_to_js(wasmHandle);
  
  // Check if we can return the buffer directly
  if (uint8Array.byteOffset === 0 && uint8Array.byteLength === uint8Array.buffer.byteLength) {
    return uint8Array.buffer;
  }
  
  // Only slice if necessary
  return uint8Array.buffer.slice(uint8Array.byteOffset, uint8Array.byteOffset + uint8Array.byteLength);
}
```

**Test to Verify Fix**:
```typescript
describe('Fixed Memory Efficiency', () => {
  it('should allocate buffers efficiently', async () => {
    const device = await WASMDevice.create();
    
    // Should not allocate JS buffer
    const data = device.createData(1024);
    expect(data).toBeDefined();
    
    // Should work correctly
    const view = device.readDataView(data, float32);
    view[0] = 42;
    expect(view[0]).toBe(42);
  });
});
```

---

### 5. Memory Pressure Handling

**Severity**: 🟡 HIGH - Stability

**Finding**: No memory limits, pools grow unbounded, manual compaction only.

**Key Files**:
- `wasm/src/memory.rs`: Lines 109-132, 376-395
- `src/device.ts`: Lines 396-406

**Problem Code**:
```rust
// wasm/src/memory.rs:109-132
fn get_buffer(&mut self) -> *mut u8 {
    if let Some(ptr) = self.available_buffers.pop() {
        ptr
    } else {
        // No limit check!
        let ptr = unsafe { std::alloc::alloc(layout) };
        // ...
    }
}
```

**Test to Demonstrate Issue**:
```typescript
// test/memory-pressure.test.ts
describe('Memory Pressure', () => {
  it('should handle memory limits gracefully', async () => {
    const device = await WASMDevice.create();
    
    // Try to allocate more than reasonable
    const allocations: DeviceData[] = [];
    const maxMemory = 256 * 1024 * 1024; // 256MB
    const chunkSize = 1 * 1024 * 1024; // 1MB chunks
    
    let allocated = 0;
    let failed = false;
    
    try {
      while (allocated < maxMemory * 2) { // Try to allocate 2x limit
        allocations.push(device.createData(chunkSize));
        allocated += chunkSize;
      }
    } catch (e) {
      failed = true;
    }
    
    expect(failed).toBe(true); // Should fail at some point
    expect(allocated).toBeLessThan(maxMemory * 1.5); // Should not allocate too much
    
    // Cleanup
    allocations.forEach(d => device.disposeData(d));
  });

  it('should automatically compact pools under pressure', async () => {
    const device = await WASMDevice.create();
    
    // Allocate and free many buffers
    for (let i = 0; i < 100; i++) {
      const data = device.createData(1024 * 1024);
      device.disposeData(data);
    }
    
    const stats1 = device.getMemoryStats();
    
    // Simulate memory pressure
    // Currently requires manual call - should be automatic
    device.performIntensiveCleanup();
    
    const stats2 = device.getMemoryStats();
    expect(stats2.totalAllocated).toBeLessThan(stats1.totalAllocated);
  });
});
```

**Suggested Fix**:

1. Add memory limits and automatic compaction:
```rust
// wasm/src/memory.rs
pub struct WasmMemoryManager {
    pools: Vec<BufferPool>,
    active_buffers: HashMap<BufferId, Arc<BufferInfo>>,
    memory_limit: usize,
    compact_threshold: f32, // Compact when usage > threshold
}

impl WasmMemoryManager {
    pub fn new_with_limit(memory_limit: usize) -> Self {
        WasmMemoryManager {
            // ...
            memory_limit,
            compact_threshold: 0.8, // Compact at 80% usage
        }
    }
    
    fn check_memory_pressure(&mut self) {
        let usage = TOTAL_ALLOCATED.load(Ordering::Relaxed);
        if usage as f32 > self.memory_limit as f32 * self.compact_threshold {
            self.compact_pools();
        }
    }
    
    fn get_buffer(&mut self, size_class: BufferSizeClass) -> Result<*mut u8, String> {
        self.check_memory_pressure();
        
        let size = size_class.actual_size();
        let current = TOTAL_ALLOCATED.load(Ordering::Relaxed);
        
        if current + size > self.memory_limit {
            // Try compaction first
            self.compact_pools();
            let new_current = TOTAL_ALLOCATED.load(Ordering::Relaxed);
            
            if new_current + size > self.memory_limit {
                return Err(format!(
                    "Memory limit exceeded: {} + {} > {}",
                    new_current, size, self.memory_limit
                ));
            }
        }
        
        // ... existing allocation code ...
    }
}
```

2. Add memory monitoring to TypeScript:
```typescript
// src/device.ts
interface MemoryConfig {
  maxMemory?: number;
  compactThreshold?: number;
  autoCompact?: boolean;
}

class WASMDevice {
  private memoryConfig: MemoryConfig;
  private lastCompactTime = 0;
  private compactInterval = 60000; // 1 minute
  
  constructor(config: MemoryConfig = {}) {
    this.memoryConfig = {
      maxMemory: config.maxMemory || 256 * 1024 * 1024, // 256MB default
      compactThreshold: config.compactThreshold || 0.8,
      autoCompact: config.autoCompact !== false,
    };
  }
  
  private checkMemoryPressure(): void {
    if (!this.memoryConfig.autoCompact) return;
    
    const stats = this.getMemoryStats();
    const usage = stats.totalAllocated / this.memoryConfig.maxMemory!;
    const now = Date.now();
    
    if (usage > this.memoryConfig.compactThreshold! && 
        now - this.lastCompactTime > this.compactInterval) {
      this.performIntensiveCleanup();
      this.lastCompactTime = now;
    }
  }
  
  createData(byteLength: number): DeviceData {
    this.checkMemoryPressure();
    // ... rest of method
  }
}
```

**Test to Verify Fix**:
```typescript
describe('Fixed Memory Pressure', () => {
  it('should respect memory limits', async () => {
    const device = await WASMDevice.create({
      maxMemory: 10 * 1024 * 1024, // 10MB limit
      autoCompact: true
    });
    
    const allocations: DeviceData[] = [];
    
    // Should fail when exceeding limit
    expect(() => {
      for (let i = 0; i < 20; i++) {
        allocations.push(device.createData(1024 * 1024)); // 1MB each
      }
    }).toThrow(/Memory limit exceeded/);
    
    // Should have allocated less than limit
    const stats = device.getMemoryStats();
    expect(stats.totalAllocated).toBeLessThanOrEqual(10 * 1024 * 1024);
  });
});
```

---

### 6. Buffer Handle Mutation Safety

**Severity**: 🟡 HIGH - API Design

**Finding**: Direct mutation of private properties bypasses encapsulation.

**Key Files**:
- `src/device.ts`: Line 361
- `src/data.ts`: Lines 23, 79

**Problem Code**:
```typescript
// src/device.ts:361
(wasmData as any).wasmHandle = newHandle;  // Bad practice!
```

**Test to Demonstrate Issue**:
```typescript
// test/encapsulation.test.ts
describe('Buffer Handle Encapsulation', () => {
  it('should not allow direct handle mutation', async () => {
    const device = await WASMDevice.create();
    const data = device.createData(1024);
    
    // This should not be possible
    expect(() => {
      (data as any).wasmHandle = null;
    }).toThrow('Cannot set property wasmHandle');
    
    // Should use proper API
    const buffer = new ArrayBuffer(1024);
    await device.writeData(data, buffer);
    
    // Data should still be valid
    const readBack = await device.readData(data);
    expect(readBack.byteLength).toBe(1024);
  });
});
```

**Suggested Fix**:

1. Add proper encapsulation to WASMDeviceData:
```typescript
// src/data.ts
export class WASMDeviceData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  readonly byteLength: number;

  #wasmHandle: unknown;  // Private field
  #wasmModule: WASMModule;
  #disposed = false;
  #cleanupToken?: object;

  constructor(device: Device, byteLength: number, wasmHandle: unknown, wasmModule: WASMModule) {
    this.device = device;
    this.byteLength = byteLength;
    this.#wasmHandle = wasmHandle;
    this.#wasmModule = wasmModule;
    this.id = `wasm-data-${(wasmHandle as any).id}`;

    // ... finalization registry setup ...
  }

  getWASMHandle(): unknown {
    if (this.#disposed) {
      throw new Error('WASMDeviceData has been disposed');
    }
    return this.#wasmHandle;
  }

  updateHandle(newHandle: unknown): void {
    if (this.#disposed) {
      throw new Error('Cannot update handle of disposed WASMDeviceData');
    }
    
    // Clean up old handle if needed
    const device = this.device as any;
    if (this.#wasmHandle && device.operationDispatcher) {
      device.operationDispatcher.release_buffer(this.#wasmHandle);
    }
    
    this.#wasmHandle = newHandle;
  }

  // ... rest of class ...
}
```

2. Update device.ts to use proper API:
```typescript
// src/device.ts
async writeData(data: DeviceData, buffer: ArrayBuffer): Promise<void> {
  // ... validation ...
  
  const wasmData = data as WASMDeviceData;
  const sourceData = new Uint8Array(buffer);
  const newHandle = this.operationDispatcher.create_buffer_with_js_data(sourceData);
  
  // Use proper API instead of direct mutation
  wasmData.updateHandle(newHandle);
}
```

**Test to Verify Fix**:
```typescript
describe('Fixed Buffer Handle Encapsulation', () => {
  it('should properly encapsulate buffer handles', async () => {
    const device = await WASMDevice.create();
    const data = device.createData(1024);
    
    // Cannot access private fields
    expect((data as any).#wasmHandle).toBeUndefined();
    
    // Can use public API
    const handle = (data as WASMDeviceData).getWASMHandle();
    expect(handle).toBeDefined();
    
    // Write updates handle properly
    await device.writeData(data, new ArrayBuffer(1024));
    const newHandle = (data as WASMDeviceData).getWASMHandle();
    expect(newHandle).toBeDefined();
  });
});
```

---

## Medium Priority Issues

### 7. Error Handling Improvements

**Severity**: 🟠 MEDIUM - Debugging/Maintenance

**Finding**: Warnings instead of errors, empty catch blocks, inconsistent error propagation.

**Key Files**:
- `src/device.ts`: Line 274
- `src/data.ts`: Line 14

**Problem Code**:
```typescript
// src/device.ts:274
if (!released) {
  console.warn(`Failed to release buffer ${wasmData.id}`);  // Should throw!
}

// src/data.ts:14
} catch {}  // Swallows all errors!
```

**Suggested Fix**:
```typescript
// Better error handling
if (!released) {
  throw new Error(`Failed to release buffer ${wasmData.id} - possible memory corruption`);
}

// Log errors for debugging
} catch (error) {
  console.error('FinalizationRegistry cleanup failed:', error);
  // Still continue - this is GC cleanup
}
```

---

### 8. Reference Counting Coordination

**Severity**: 🟠 MEDIUM - Memory Management

**Finding**: Each JS clone creates separate FinalizationRegistry entry.

**Key Files**:
- `src/data.ts`: Lines 42-51
- `wasm/src/operations/mod.rs`: Lines 279-282

**Suggested Fix**:
```typescript
// Track clones properly
class WASMDeviceData {
  #refCount = 1;
  #isClone = false;
  
  clone(): WASMDeviceData {
    // ... increment WASM ref count ...
    const cloned = new WASMDeviceData(...);
    cloned.#isClone = true;
    this.#refCount++;
    return cloned;
  }
  
  dispose(): void {
    this.#refCount--;
    if (this.#refCount === 0) {
      // Actually dispose
    }
  }
}
```

---

## Architecture Improvements

### 9. Abstraction Layers

**Severity**: 🟢 LOW - Architecture

**Finding**: Tight coupling between layers, missing abstractions.

**Suggested Improvements**:

1. **Buffer Pool Interface**:
```typescript
interface BufferPool {
  allocate(size: number): BufferHandle;
  release(handle: BufferHandle): void;
  compact(): void;
  getStats(): PoolStats;
}
```

2. **Memory Manager Interface**:
```typescript
interface MemoryManager {
  createBuffer(size: number): Buffer;
  createView(buffer: Buffer, offset: number, length: number): BufferView;
  dispose(buffer: Buffer): void;
  getMemoryUsage(): MemoryStats;
}
```

3. **View Manager Interface**:
```typescript
interface ViewManager {
  createView(buffer: Buffer, dtype: DType): TypedArrayView;
  invalidateBuffer(bufferId: string): void;
  isViewValid(view: TypedArrayView): boolean;
}
```

---

## Testing Strategy

### Phase 1: Failing Tests (Current State)
1. Memory safety tests - demonstrate use-after-free
2. View operation tests - show silent failures
3. Bounds checking tests - expose out-of-bounds access
4. Memory efficiency tests - count unnecessary allocations
5. Memory pressure tests - show unbounded growth

### Phase 2: Implementation
1. Fix critical safety issues first
2. Implement missing operations
3. Add proper error handling
4. Optimize memory usage
5. Add abstractions

### Phase 3: Passing Tests (Fixed State)
1. All safety tests pass with proper errors
2. View operations work correctly
3. Bounds are properly checked
4. Minimal memory allocations
5. Memory limits enforced

### Performance Benchmarks
```typescript
// benchmark/memory.bench.ts
describe('Memory Performance', () => {
  bench('buffer allocation', () => {
    const device = await WASMDevice.create();
    const start = performance.now();
    
    for (let i = 0; i < 1000; i++) {
      const data = device.createData(1024);
      device.disposeData(data);
    }
    
    const end = performance.now();
    expect(end - start).toBeLessThan(100); // < 100ms for 1000 allocations
  });

  bench('zero-copy views', () => {
    const device = await WASMDevice.create();
    const data = device.createData(1024 * 1024); // 1MB
    
    const start = performance.now();
    
    for (let i = 0; i < 10000; i++) {
      const view = device.readDataView(data, float32);
      view[0]; // Access to ensure view is created
    }
    
    const end = performance.now();
    expect(end - start).toBeLessThan(10); // < 10ms for 10000 views
  });
});
```

---

## Implementation Order

1. **Week 1**: Critical Safety Issues
   - View lifetime management
   - Bounds checking
   - Type safety

2. **Week 2**: Core Functionality
   - Implement view operations
   - Fix error handling
   - Add proper encapsulation

3. **Week 3**: Performance
   - Remove unnecessary allocations
   - Add memory limits
   - Optimize buffer pools

4. **Week 4**: Architecture
   - Add abstraction layers
   - Improve API design
   - Documentation

---

## Success Metrics

1. **Safety**: Zero memory corruption bugs
2. **Performance**: 50% reduction in allocations
3. **Reliability**: All operations properly implemented
4. **Maintainability**: Clear abstractions and error messages
5. **Testing**: 100% coverage of critical paths

---

## Notes

- All code snippets are simplified for clarity
- Full implementations should include proper error handling
- Performance numbers are targets, not guarantees
- Some fixes may require breaking API changes