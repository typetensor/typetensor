# @typetensor/einops

## 0.3.0

### Minor Changes

- Refactor einops operations into dedicated package

  Einops operations have been moved from `@typetensor/core` to a new dedicated `@typetensor/einops` package.

  This change improves modularity and allows users to import only the einops functionality they need. The einops operations now include comprehensive caching for better performance.

  **Migration:**
  - Install the new package: `npm install @typetensor/einops`
  - Update imports: `import { rearrange, reduce, repeat } from '@typetensor/einops'`
  - The API remains the same, only the import source has changed

  **New features:**
  - Added caching for rearrange, reduce, and repeat operations
  - Improved einops performance through optimized execution paths
  - Better separation of concerns between core tensor operations and einops

### Patch Changes

- Updated dependencies
  - @typetensor/core@0.3.0
