export class CUDABackend {
  static async create(): Promise<CUDABackend> {
    throw new Error('CUDA backend is not yet implemented');
  }
}