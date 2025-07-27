#\!/bin/bash

update_backend() {
  local backend=$1
  local desc=$2
  local class_name=$3
  local requirements=$4
  
  # Update README
  cat > packages/backend-$backend/README.md << README_EOF
# @typetensor/backend-$backend

$desc backend for TypedTensor.

## Installation

\`\`\`bash
npm install @typetensor/backend-$backend
\`\`\`

## Features

- Hardware-accelerated tensor operations using $desc
- Optimized kernels for common operations
- Efficient memory management
- Support for all core tensor operations

## Usage

\`\`\`typescript
import { tensor } from '@typetensor/core';
import { ${class_name}Backend } from '@typetensor/backend-$backend';

// Initialize $desc backend
const backend = await ${class_name}Backend.create();

// Create tensors that will run on $desc
const a = tensor([1, 2, 3, 4], { shape: [2, 2], backend });
const b = tensor([5, 6, 7, 8], { shape: [2, 2], backend });

// Operations run on $desc hardware
const result = a.add(b);
\`\`\`

## Requirements

$requirements
- TypeScript 5.9.0 or higher

## Status

⚠️ This package is currently a placeholder and under development.

## License

MIT
README_EOF

  # Update index.ts
  cat > packages/backend-$backend/src/index.ts << INDEX_EOF
export class ${class_name}Backend {
  static async create(): Promise<${class_name}Backend> {
    throw new Error('$desc backend is not yet implemented');
  }
}
INDEX_EOF
}

# Update CUDA backend
update_backend "cuda" "NVIDIA CUDA" "CUDA" "- NVIDIA GPU with CUDA support\n- CUDA runtime and drivers"

# Update Metal backend  
update_backend "metal" "Apple Metal" "Metal" "- Apple Silicon Mac or iOS device\n- macOS 11+ or iOS 14+"

# Update Vulkan backend
update_backend "vulkan" "Vulkan" "Vulkan" "- GPU with Vulkan support\n- Vulkan drivers installed"

