/**
 * Shader caching system for WebGPU backend
 * 
 * Caches compiled compute pipelines to avoid recompilation
 */

/**
 * Cache entry for a compiled pipeline
 */
interface CacheEntry {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  lastUsed: number;
}

/**
 * Shader cache manager
 */
export class ShaderCache {
  private cache = new Map<string, CacheEntry>();
  private maxCacheSize = 1000;
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Get a cached pipeline or create a new one
   */
  async getOrCreatePipeline(
    key: string,
    shaderCode: string,
    bindGroupLayoutDescriptor?: GPUBindGroupLayoutDescriptor,
  ): Promise<{ pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout }> {
    // Check cache
    const cached = this.cache.get(key);
    if (cached) {
      cached.lastUsed = Date.now();
      return { pipeline: cached.pipeline, bindGroupLayout: cached.bindGroupLayout };
    }

    // Create shader module
    const shaderModule = this.device.createShaderModule({
      label: `Shader: ${key}`,
      code: shaderCode,
    });

    // Create bind group layout
    const bindGroupLayout = bindGroupLayoutDescriptor
      ? this.device.createBindGroupLayout(bindGroupLayoutDescriptor)
      : this.device.createBindGroupLayout({
          entries: this.inferBindGroupLayoutEntries(shaderCode),
        });

    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      label: `Pipeline Layout: ${key}`,
      bindGroupLayouts: [bindGroupLayout],
    });

    // Create compute pipeline
    const pipeline = await this.device.createComputePipelineAsync({
      label: `Pipeline: ${key}`,
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    // Cache the pipeline
    this.cache.set(key, {
      pipeline,
      bindGroupLayout,
      lastUsed: Date.now(),
    });

    // Evict old entries if cache is too large
    this.evictIfNeeded();

    return { pipeline, bindGroupLayout };
  }

  /**
   * Clear the cache
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getStats(): { size: number; maxSize: number } {
    return {
      size: this.cache.size,
      maxSize: this.maxCacheSize,
    };
  }

  /**
   * Infer bind group layout entries from shader code
   * This is a simple implementation - could be enhanced with proper WGSL parsing
   */
  private inferBindGroupLayoutEntries(shaderCode: string): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [];
    
    // Match @group(0) @binding(N) patterns
    const bindingPattern = /@group\(0\)\s+@binding\((\d+)\)\s+var<(storage|uniform)/g;
    let match;
    
    while ((match = bindingPattern.exec(shaderCode)) !== null) {
      const bindingStr = match[1];
      const storageType = match[2];
      
      if (!bindingStr || !storageType) continue;
      
      const binding = parseInt(bindingStr, 10);
      
      // Check if it's read-only or read-write
      const isReadOnly = shaderCode.includes(`@binding(${binding}) var<${storageType}, read>`);
      
      entries.push({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: storageType === 'uniform' ? 'uniform' : 
                isReadOnly ? 'read-only-storage' : 'storage',
        },
      });
    }

    return entries;
  }

  /**
   * Evict least recently used entries if cache is too large
   */
  private evictIfNeeded(): void {
    if (this.cache.size <= this.maxCacheSize) {
      return;
    }

    // Sort entries by last used time
    const entries = Array.from(this.cache.entries())
      .sort((a, b) => a[1].lastUsed - b[1].lastUsed);

    // Remove oldest entries
    const toRemove = entries.slice(0, this.cache.size - this.maxCacheSize * 0.9);
    for (const [key] of toRemove) {
      this.cache.delete(key);
    }
  }
}