
export { WASMDevice } from './device';
export { WASMDeviceData } from './data';
export { loadWASMModule } from './loader';
export * from './types';
export { MemoryViewManager } from './memory-views';
export type { MemoryView } from './memory-views';

import { WASMDevice } from './device';
let _wasmDevice: WASMDevice | null = null;

export async function getWASMDevice(): Promise<WASMDevice> {
  if (!_wasmDevice) {
    _wasmDevice = await WASMDevice.create();
  }
  return _wasmDevice;
}

export async function createWASMDevice(): Promise<WASMDevice> {
  return await WASMDevice.create();
}

export class WASMBackend {
  static async create(): Promise<WASMDevice> {
    return await getWASMDevice();
  }
}
