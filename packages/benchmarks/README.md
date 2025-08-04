# @typetensor/benchmarks

Internal benchmarking suite for TypeTensor operations. This package provides comprehensive benchmarks for tensor operations across different backends, starting with the CPU backend.

## Overview

The benchmarks are designed to:
- Test individual tensor operations in isolation
- Measure performance across different tensor sizes and shapes
- Compare performance between different backends
- Track performance regressions over time

## Structure

```
benchmarks/
├── src/
│   ├── operations/      # Individual operation benchmarks
│   │   └── creation.bench.ts
│   ├── runners/         # Standalone benchmark runners
│   │   └── tensor-creation.ts
│   └── utils/          # Benchmark utilities
│       ├── sizes.ts    # Common tensor sizes
│       ├── data.ts     # Data generation
│       └── formatting.ts # Result formatting
```

## Running Benchmarks

### Using Vitest (Recommended for CI)

```bash
# Run all benchmarks
bun run bench

# Watch mode
bun run bench:watch

# Run specific benchmark file
bun run bench creation
```

### Using Standalone Runners

```bash
# Run tensor creation benchmarks
bun run bench:creation
```

## Writing New Benchmarks

### 1. Using Vitest Bench

```typescript
import { bench, describe } from 'vitest';
import { tensor, float32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { MATRIX_SIZES } from '../utils/sizes';

describe('matrix operations', () => {
  for (const size of MATRIX_SIZES) {
    bench(`multiply ${size.name}`, async () => {
      const a = await tensor(data, { device: cpu, dtype: float32 });
      const b = await tensor(data, { device: cpu, dtype: float32 });
      await a.matmul(b);
    });
  }
});
```

### 2. Using Tinybench Directly

```typescript
import { Bench } from 'tinybench';
import { formatBenchResults } from '@typetensor/benchmarks';

const bench = new Bench({ time: 500 });

bench.add('operation name', async () => {
  // benchmark code
});

await bench.run();
const results = formatBenchResults(bench);
```

## Benchmark Utilities

### Size Presets

```typescript
import { VECTOR_SIZES, MATRIX_SIZES, TENSOR_3D_SIZES } from '@typetensor/benchmarks';

// Predefined sizes: tiny, small, medium, large, xlarge
// Each with shape and element count
```

### Data Generation

```typescript
import { generateRandomData, generateSequentialData } from '@typetensor/benchmarks';

const randomData = generateRandomData([100, 100]);
const { data: seqData } = generateSequentialData([100, 100]);
```

### Result Formatting

```typescript
import { formatPerformanceSummary, resultsToMarkdownTable } from '@typetensor/benchmarks';

const summary = formatPerformanceSummary(results);
const table = resultsToMarkdownTable(results);
```

## Continuous Benchmarking

For tracking performance over time:

1. **CodSpeed** (Recommended for Vitest)
   - Install: `bun add -d @codspeed/vitest-plugin`
   - Integrates directly with Vitest bench
   - Provides <1% variance in CI

2. **Bencher**
   - General purpose continuous benchmarking
   - Works with any benchmark output format

3. **GitHub Actions**
   - Use `benchmark-action/github-action-benchmark`
   - Stores results in GitHub Pages

## Performance Considerations

- Benchmarks run with warmup phase by default
- Each benchmark runs for at least 500ms for statistical significance
- Results include ops/sec, mean, p50, p95, p99, standard deviation
- Use dedicated CI runners when possible to reduce variance