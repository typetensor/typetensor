import torch
import numpy as np

print("=== PyTorch Behavior Research ===")
print()

# Test the exact scenario from our failing test
print("1. Basic reshape -> transpose -> flatten chain:")
original = torch.tensor([1, 2, 3, 4, 5, 6])
print(f"Original: {original}")
print(f"Original shape: {original.shape}")
print(f"Original strides: {original.stride()}")
print(f"Original is_contiguous: {original.is_contiguous()}")
print()

reshaped = original.reshape(2, 3)
print(f"After reshape(2,3): {reshaped}")
print(f"Reshaped shape: {reshaped.shape}")
print(f"Reshaped strides: {reshaped.stride()}")
print(f"Reshaped is_contiguous: {reshaped.is_contiguous()}")
print(f"Reshaped shares memory with original: {reshaped.data_ptr() == original.data_ptr()}")
print()

transposed = reshaped.transpose(0, 1)  # or reshaped.T
print(f"After transpose: {transposed}")
print(f"Transposed shape: {transposed.shape}")
print(f"Transposed strides: {transposed.stride()}")
print(f"Transposed is_contiguous: {transposed.is_contiguous()}")
print(f"Transposed shares memory with original: {transposed.data_ptr() == original.data_ptr()}")
print()

flattened = transposed.flatten()
print(f"After flatten: {flattened}")
print(f"Flattened shape: {flattened.shape}")
print(f"Flattened strides: {flattened.stride()}")
print(f"Flattened is_contiguous: {flattened.is_contiguous()}")
print(f"Flattened shares memory with original: {flattened.data_ptr() == original.data_ptr()}")
print()

print("2. Direct chain (one-liner):")
chained = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3).transpose(0, 1).flatten()
print(f"Chained result: {chained}")
print(f"Chained is_contiguous: {chained.is_contiguous()}")
print()

print("3. Testing different flatten methods:")
transposed_again = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3).transpose(0, 1)
print(f"Before flatten - is_contiguous: {transposed_again.is_contiguous()}")

# Try different flatten approaches
flatten_result = transposed_again.flatten()
print(f"flatten(): {flatten_result}")
print(f"flatten() shares memory: {flatten_result.data_ptr() == transposed_again.data_ptr()}")

try:
    view_result = transposed_again.view(-1)
    print(f"view(-1): {view_result}")
    print(f"view(-1) shares memory: {view_result.data_ptr() == transposed_again.data_ptr()}")
except RuntimeError as e:
    print(f"view(-1) failed: {e}")

reshape_result = transposed_again.reshape(-1)
print(f"reshape(-1): {reshape_result}")
print(f"reshape(-1) shares memory: {reshape_result.data_ptr() == transposed_again.data_ptr()}")
print()

print("4. NumPy comparison:")
np_original = np.array([1, 2, 3, 4, 5, 6])
np_reshaped = np_original.reshape(2, 3)
np_transposed = np_reshaped.T
np_flattened = np_transposed.flatten()
print(f"NumPy result: {np_flattened}")
print(f"NumPy shares memory: {np.shares_memory(np_flattened, np_original)}")
print()

print("5. Testing contiguous() method:")
made_contiguous = transposed_again.contiguous()
print(f"After contiguous(): {made_contiguous}")
print(f"contiguous() result: {made_contiguous.flatten()}")
print(f"contiguous() shares memory with transposed: {made_contiguous.data_ptr() == transposed_again.data_ptr()}")