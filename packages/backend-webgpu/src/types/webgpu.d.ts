/// <reference types="@webgpu/types" />

declare module 'webgpu' {
  export class GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }
}