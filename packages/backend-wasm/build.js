#!/usr/bin/env node

/**
 * Build script for TypeTensor WASM backend
 * 
 * This script handles the complete build process:
 * 1. Builds Rust code to WebAssembly using wasm-pack
 * 2. Processes the generated TypeScript bindings
 * 3. Creates optimized bundles for different targets
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const WASM_DIR = path.join(__dirname, 'wasm');
const PKG_DIR = path.join(__dirname, 'pkg');
const SRC_DIR = path.join(__dirname, 'src');

function log(message) {
  console.log(`[build] ${message}`);
}

function run(command, options = {}) {
  log(`Running: ${command}`);
  try {
    execSync(command, { 
      stdio: 'inherit', 
      cwd: options.cwd || __dirname,
      ...options 
    });
  } catch (error) {
    console.error(`Failed to run: ${command}`);
    process.exit(1);
  }
}

function main() {
  log('Building TypeTensor WASM backend...');
  
  // Check if wasm-pack is installed
  try {
    execSync('wasm-pack --version', { stdio: 'pipe' });
  } catch (error) {
    console.error('wasm-pack is not installed. Please install it first:');
    console.error('curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh');
    process.exit(1);
  }

  // Clean previous build
  if (fs.existsSync(PKG_DIR)) {
    log('Cleaning previous build...');
    fs.rmSync(PKG_DIR, { recursive: true, force: true });
  }

  // Build with wasm-pack
  log('Building Rust to WebAssembly...');
  run('wasm-pack build --target bundler --out-dir ../pkg --scope typetensor', {
    cwd: WASM_DIR
  });

  // Post-process the generated files
  log('Post-processing generated files...');
  
  // Read and modify package.json
  const pkgJsonPath = path.join(PKG_DIR, 'package.json');
  if (fs.existsSync(pkgJsonPath)) {
    const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
    
    // Update package metadata
    pkgJson.name = '@typetensor/backend-wasm-core';
    pkgJson.description = 'WebAssembly core for TypeTensor - High-performance tensor operations';
    pkgJson.keywords = ['webassembly', 'wasm', 'tensor', 'ml', 'linear-algebra', 'simd'];
    pkgJson.repository = {
      type: 'git',
      url: 'https://github.com/typetensor/typetensor.git',
      directory: 'packages/backend-wasm'
    };
    pkgJson.bugs = 'https://github.com/typetensor/typetensor/issues';
    pkgJson.homepage = 'https://github.com/typetensor/typetensor#readme';
    
    fs.writeFileSync(pkgJsonPath, JSON.stringify(pkgJson, null, 2));
    log('Updated package.json');
  }

  // Create README for the generated package
  const readmePath = path.join(PKG_DIR, 'README.md');
  const readmeContent = `# @typetensor/backend-wasm-core

WebAssembly core module for TypeTensor backend.

This package contains the compiled WebAssembly module and TypeScript bindings.
Use \`@typetensor/backend-wasm\` instead for the complete backend implementation.

## Features

- High-performance tensor operations compiled to WebAssembly
- SIMD optimizations where supported
- Memory-efficient operations with custom allocators
- Support for multiple data types (f32, f64, i32, etc.)

## Usage

This is a low-level package. Use \`@typetensor/backend-wasm\` for the complete API.

\`\`\`typescript
import * as wasm from '@typetensor/backend-wasm-core';

// Initialize the WASM module
await wasm.default();

// Use the WASM functions...
\`\`\`
`;
  
  fs.writeFileSync(readmePath, readmeContent);
  log('Created README.md');

  log('Build completed successfully!');
  log(`Generated files in: ${PKG_DIR}`);
  
  // Show build statistics
  const wasmFiles = fs.readdirSync(PKG_DIR).filter(f => f.endsWith('.wasm'));
  for (const wasmFile of wasmFiles) {
    const filePath = path.join(PKG_DIR, wasmFile);
    const stats = fs.statSync(filePath);
    const sizeKb = (stats.size / 1024).toFixed(1);
    log(`${wasmFile}: ${sizeKb} KB`);
  }
}

if (require.main === module) {
  main();
}