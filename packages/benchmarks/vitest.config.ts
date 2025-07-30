import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['**/*.bench.ts'],
    benchmark: {
      include: ['**/*.bench.ts'],
    },
  },
});