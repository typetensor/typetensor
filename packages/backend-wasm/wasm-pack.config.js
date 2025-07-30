/**
 * wasm-pack configuration for TypeTensor WASM backend
 * 
 * Optimizes the build for production use with minimal bundle size
 * and maximum performance.
 */

module.exports = {
  // Build target - we want to support both browsers and Node.js
  target: 'bundler',
  
  // Output directory
  outDir: 'pkg',
  
  // Scope for npm package
  scope: '@typetensor',
  
  // Build profile - use release for production
  profile: 'release',
  
  // Enable wasm-opt optimizations
  wasmOpt: [
    // Size optimizations
    '-Os',
    '--enable-bulk-memory',
    '--enable-sign-ext',
    '--enable-mutable-globals',
    '--enable-nontrapping-float-to-int',
    
    // SIMD optimizations (where supported)
    '--enable-simd',
  ],
  
  // Generate TypeScript definitions
  typescript: true,
  
  // Additional cargo build arguments
  cargoArgs: ['--features', 'console_error_panic_hook'],
  
  // Extra files to include in the package
  extraFiles: [
    'README.md',
    '../LICENSE',
  ],
};