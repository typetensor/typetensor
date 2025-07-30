/**
 * Benchmark configuration utilities for statistical reliability
 */

import type { BenchOptions } from 'tinybench';

/**
 * Configuration profiles for different benchmark scenarios
 */
export const BENCHMARK_PROFILES = {
  /**
   * Quick profile for development/debugging
   * Faster but less reliable results
   */
  quick: {
    time: 250,
    iterations: 10,
    warmup: true,
  } as BenchOptions,

  /**
   * Standard profile for regular benchmarking
   * Good balance of speed and reliability
   */
  standard: {
    time: 1000,
    warmup: true,
  } as BenchOptions,

  /**
   * High-precision profile for CI/production
   * Longer runtime but very reliable results
   */
  precise: {
    time: 2000,
    iterations: 200,
    warmup: true,
    setup(task: any, mode: string) {
      if (mode === 'warmup') {
        // Force garbage collection multiple times
        if (global.gc) {
          global.gc();
          global.gc();
          global.gc();
        }
      }
    },
    teardown() {
      // Clean up after each benchmark
      if (global.gc) {
        global.gc();
      }
    },
  } as BenchOptions,
} as const;

/**
 * Get benchmark configuration based on environment
 */
export function getBenchmarkConfig(): BenchOptions {
  const profile = process.env.BENCHMARK_PROFILE || 'standard';

  switch (profile) {
    case 'quick':
      return BENCHMARK_PROFILES.quick;
    case 'precise':
      return BENCHMARK_PROFILES.precise;
    case 'standard':
    default:
      return BENCHMARK_PROFILES.standard;
  }
}

/**
 * Statistical analysis helpers
 */
export function analyzeResults(samples: number[]): {
  mean: number;
  median: number;
  stdDev: number;
  cv: number; // Coefficient of variation
  outliers: number;
  isStable: boolean;
} {
  const sorted = [...samples].sort((a, b) => a - b);
  const n = samples.length;

  // Mean
  const mean = samples.reduce((sum, val) => sum + val, 0) / n;

  // Median
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];

  // Standard deviation
  const variance = samples.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
  const stdDev = Math.sqrt(variance);

  // Coefficient of variation (relative standard deviation)
  const cv = stdDev / mean;

  // Outlier detection using IQR method
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;
  const outliers = samples.filter((val) => val < lowerBound || val > upperBound).length;

  // Consider stable if CV < 5% and outliers < 5%
  const isStable = cv < 0.05 && outliers / n < 0.05;

  return {
    mean,
    median,
    stdDev,
    cv,
    outliers,
    isStable,
  };
}

/**
 * Recommendations for benchmark reliability
 */
export function getBenchmarkRecommendations(): string[] {
  const recommendations = [
    '🔧 For best results, run benchmarks on a dedicated machine',
    '🔋 Ensure stable power supply (avoid battery mode)',
    '🌡️  Allow system to warm up and reach thermal equilibrium',
    '🔇 Close unnecessary applications to reduce system noise',
    '⚡ Consider using Node.js with --expose-gc flag for garbage collection control',
    '📊 Run multiple benchmark sessions and compare results',
    '⏰ Be aware that results may vary between different times of day',
  ];

  if (process.env.NODE_ENV === 'development') {
    recommendations.push('🚀 Use BENCHMARK_PROFILE=quick for faster development cycles');
  }

  if (process.env.CI) {
    recommendations.push(
      '🏗️  CI environments may have higher variance - consider dedicated runners',
    );
  }

  return recommendations;
}
