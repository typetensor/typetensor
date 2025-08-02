export class WebGPUBackend {
  static async create(): Promise<WebGPUBackend> {
    throw new Error('WebGPU backend is not yet implemented');
  }
}
