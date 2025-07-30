// Test Bun's FFI capabilities
console.log('Testing Bun FFI...');

try {
  const { dlopen, FFIType, suffix } = await import('bun:ffi');
  
  console.log('Bun FFI module loaded successfully!');
  console.log('Available FFI types:', Object.keys(FFIType));
  console.log('Library suffix for this platform:', suffix);
  
  // Try to load a system library (libm - math library)
  const libm = dlopen(`libm.${suffix}`, {
    cosf: {
      args: [FFIType.f32],
      returns: FFIType.f32,
    },
    sinf: {
      args: [FFIType.f32], 
      returns: FFIType.f32,
    },
  });
  
  console.log('System math library loaded successfully!');
  
  // Test calling a function
  const result = libm.symbols.cosf(0);
  console.log('cos(0) =', result); // Should be 1.0
  
} catch (error) {
  console.error('FFI test failed:', error);
}