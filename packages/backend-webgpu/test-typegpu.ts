import tgpu from 'typegpu';

// Test if we can use TypeGPU
async function testTypeGPU() {
  try {
    console.log('Testing TypeGPU initialization...');

    // Initialize TypeGPU root
    const root = await tgpu.init();
    console.log('TypeGPU root initialized successfully!');

    // Test basic buffer creation
    const buffer = root.createBuffer(new Float32Array([1, 2, 3, 4])).$usage('storage');
    console.log('Buffer created successfully!');

    // Clean up
    root.destroy();
    console.log('TypeGPU test completed successfully!');
  } catch (error) {
    console.error('TypeGPU test failed:', error);
    throw error;
  }
}

testTypeGPU();
