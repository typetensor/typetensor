import { describe, test, expect, beforeAll } from 'bun:test';
import { loadWASMModule, isWASMModuleLoaded, resetWASMModule } from './loader';

describe('WASM Loader', () => {
  beforeAll(() => {
    // Reset module state before tests
    resetWASMModule();
  });

  test('should load WASM module', async () => {
    expect(isWASMModuleLoaded()).toBe(false);
    
    const module = await loadWASMModule({ debug: true });
    
    expect(module).toBeDefined();
    expect(isWASMModuleLoaded()).toBe(true);
  });

  test('should return cached module on subsequent calls', async () => {
    const module1 = await loadWASMModule();
    const module2 = await loadWASMModule();
    
    expect(module1).toBe(module2);
  });

  test('should have expected exports', async () => {
    const module = await loadWASMModule();
    
    // Core functions
    expect(typeof module.greet).toBe('function');
    expect(typeof module.get_version).toBe('function');
    
    // Classes
    expect(module.WasmOperationDispatcher).toBeDefined();
    expect(module.WasmMemoryManager).toBeDefined();
    expect(module.WasmBufferHandle).toBeDefined();
    expect(module.WasmTensorMeta).toBeDefined();
    
    // Utility functions
    expect(typeof module.has_simd_support).toBe('function');
    expect(typeof module.has_shared_memory_support).toBe('function');
    expect(typeof module.get_optimal_thread_count).toBe('function');
  });

  test('should get version', async () => {
    const module = await loadWASMModule();
    const version = module.get_version();
    
    expect(version).toBe('0.1.0');
  });
});