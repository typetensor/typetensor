# Root Cause Analysis: Transpose + Flatten Bug

## Current Behavior
- Input: `[1,2,3,4,5,6]` → reshape([2,3]) → transpose() → flatten()
- Expected: `[1,4,2,5,3,6]` (PyTorch/NumPy behavior)
- Actual: `[1,2,3,4,5,6]` (original order)

## Analysis Points to Investigate

### 1. Data Reading from Device
**Question**: When we call `toArray()` on a transposed tensor, does it read data in the correct order?

Let's trace what happens when we call `toArray()` on the transposed tensor:
- Transposed tensor has strides `[1, 3]` for shape `[3, 2]`
- Does `device.readData()` respect these strides?
- Does `bufferToNestedArray()` use the strides correctly?

### 2. Clone Implementation
**Question**: What exactly does `clone()` do with view metadata?

Current clone:
```typescript
async clone(): Promise<Tensor<S>> {
  const buffer = await this.data.device.readData(this.data);
  const newData = this.data.device.createData(buffer.byteLength);
  await this.data.device.writeData(newData, buffer);
  return new Tensor(this.transform, newData); // PRESERVES TRANSFORM!
}
```

Issues:
- Preserves non-contiguous transform metadata
- May not actually copy data in contiguous order

### 3. Device Data Reading with Strides
**Question**: How does the backend handle reading strided/non-contiguous data?

When device reads transposed data:
- Does it follow the strides correctly?
- Does it return data in the view's logical order or physical memory order?
- Are we getting the raw physical buffer instead of the logical view?

### 4. toArray() Implementation  
**Question**: Does `toArray()` properly handle non-contiguous layouts?

```typescript
async toArray() {
  const buffer = await this.data.device.readData(this.data);
  return bufferToNestedArray(buffer, this.shape, this.dtype, this.strides, this.storage.__offset);
}
```

Critical: Does `bufferToNestedArray` correctly use the strides to reconstruct the logical view?

### 5. Backend Execute vs Data Reading
**Question**: Is there a difference between how the backend executes operations vs how it reads data?

- Operations like transpose might only update metadata
- Actual data might still be in original physical layout
- Reading might not respect the view transformation

### 6. Memory Layout Assumptions
**Question**: Are we making incorrect assumptions about memory layout?

PyTorch behavior:
- Views share memory but have different logical layouts
- `flatten()` on non-contiguous creates a **new contiguous copy**
- The copy has data in the **logical order** of the view

Our current assumption:
- Views might not be properly implemented
- Data reading might not respect stride layout