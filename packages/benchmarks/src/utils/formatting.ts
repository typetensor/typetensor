/**
 * Benchmark result formatting utilities
 */

import type { Bench, Task } from 'tinybench';

export interface FormattedResult {
  name: string;
  ops: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  stdDev: number;
  margin: number;
  samples: number;
  cv: number; // Coefficient of variation
}

/**
 * Format a single benchmark task result
 */
export function formatTaskResult(name: string, task: Task): FormattedResult | null {
  const result = task.result;
  if (!result) {
    return null;
  }

  // Convert from seconds to nanoseconds
  const meanNs = (result.mean || 0) * 1_000_000_000;
  const stdDevNs = (result.sd || 0) * 1_000_000_000;
  const p75Ns = (result.p75 || 0) * 1_000_000_000; // tinybench provides p75, not p50
  const p99Ns = (result.p99 || 0) * 1_000_000_000; // this exists
  const marginNs = (result.moe || 0) * 1_000_000_000;

  const cv = meanNs > 0 ? stdDevNs / meanNs : 0; // Coefficient of variation

  return {
    name,
    ops: result.hz || 0,
    mean: meanNs,
    p50: p75Ns, // Using p75 as closest to p50
    p95: p99Ns, // Using p99 as conservative estimate for p95
    p99: p99Ns,
    stdDev: stdDevNs,
    margin: marginNs,
    samples: result.samples?.length || 0,
    cv,
  };
}

/**
 * Format all benchmark results from a Bench instance
 */
export function formatBenchResults(bench: Bench): FormattedResult[] {
  const results: FormattedResult[] = [];

  // bench.tasks is a Map, so we need to iterate properly
  // The Map key is numeric index, but task.name has the actual name
  bench.tasks.forEach((task) => {
    const formatted = formatTaskResult(task.name, task);
    if (formatted) {
      results.push(formatted);
    }
  });

  return results;
}

/**
 * Create a markdown table from benchmark results
 */
export function resultsToMarkdownTable(results: FormattedResult[]): string {
  const headers = [
    'Name',
    'Ops/sec',
    'Mean (ms)',
    'P50 (ms)',
    'P95 (ms)',
    'P99 (ms)',
    'Std Dev',
    'Margin',
  ];
  const separator = headers.map((h) => '-'.repeat(h.length));

  const rows = results.map((r) => [
    r.name,
    r.ops.toFixed(2),
    r.mean.toFixed(3),
    r.p50.toFixed(3),
    r.p95.toFixed(3),
    r.p99.toFixed(3),
    r.stdDev.toFixed(3),
    `±${r.margin.toFixed(3)}`,
  ]);

  const table = [headers.join(' | '), separator.join(' | '), ...rows.map((row) => row.join(' | '))];

  return table.join('\n');
}

/**
 * Format results for individual scenario tracking
 */
export function formatIndividualResults(results: FormattedResult[]): string {
  const lines: string[] = [];

  lines.push('## Individual Scenario Results\n');

  for (const result of results) {
    const stability =
      result.cv < 0.05 ? '🟢 Stable' : result.cv < 0.1 ? '🟡 Moderate' : '🔴 High variance';

    // Choose appropriate unit based on magnitude
    const formatLatency = (ns: number) => {
      if (ns >= 1_000_000) {
        return `${(ns / 1_000_000).toFixed(3)}ms`;
      }
      if (ns >= 1_000) {
        return `${(ns / 1_000).toFixed(1)}μs`;
      }
      return `${ns.toFixed(0)}ns`;
    };

    lines.push(`### ${result.name}`);
    lines.push(`- **Ops/sec**: ${result.ops.toFixed(2)}`);
    lines.push(`- **Mean latency**: ${formatLatency(result.mean)}`);
    lines.push(`- **P95 latency**: ${formatLatency(result.p95)}`);
    lines.push(
      `- **Samples**: ${result.samples} (CV: ${(result.cv * 100).toFixed(1)}%) ${stability}`,
    );
    lines.push(`- **Standard deviation**: ±${formatLatency(result.stdDev)}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Export results in a format suitable for continuous benchmarking
 */
export function exportForTracking(results: FormattedResult[]): Record<string, any> {
  const timestamp = new Date().toISOString();
  const data: Record<string, any> = {
    timestamp,
    commit: process.env.GITHUB_SHA || 'local',
    scenarios: {},
  };

  for (const result of results) {
    data.scenarios[result.name] = {
      ops_per_sec: result.ops,
      mean_ns: result.mean, // Changed to nanoseconds
      p50_ns: result.p50,
      p95_ns: result.p95,
      p99_ns: result.p99,
      std_dev_ns: result.stdDev,
      margin_of_error_ns: result.margin,
      samples: result.samples,
      cv: result.cv,
    };
  }

  return data;
}
