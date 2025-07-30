/**
 * Utilities for tracking benchmark results over time
 */

import { writeFileSync, readFileSync, existsSync } from 'fs';
import { join } from 'path';

export interface BenchmarkHistory {
  timestamp: string;
  commit: string;
  scenarios: Record<string, {
    ops_per_sec: number;
    mean_ns: number;
    p50_ns: number;
    p95_ns: number;
    p99_ns: number;
    std_dev_ns: number;
    margin_of_error_ns: number;
    samples: number;
    cv: number;
  }>;
}

/**
 * Save benchmark results to history file
 */
export function saveBenchmarkHistory(data: BenchmarkHistory, filepath = 'benchmark-history.json') {
  let history: BenchmarkHistory[] = [];
  
  // Load existing history if it exists
  if (existsSync(filepath)) {
    try {
      const existing = readFileSync(filepath, 'utf8');
      history = JSON.parse(existing);
    } catch (error) {
      console.warn('Could not read existing benchmark history:', error);
    }
  }
  
  // Add new entry
  history.push(data);
  
  // Save updated history
  writeFileSync(filepath, JSON.stringify(history, null, 2));
  console.log(`Saved benchmark results to ${filepath}`);
}

/**
 * Compare current results with historical data
 */
export function compareWithHistory(
  current: BenchmarkHistory,
  historyPath = 'benchmark-history.json'
): void {
  if (!existsSync(historyPath)) {
    console.log('No historical data found for comparison');
    return;
  }
  
  try {
    const historyData: BenchmarkHistory[] = JSON.parse(readFileSync(historyPath, 'utf8'));
    const previous = historyData[historyData.length - 1];
    
    if (!previous) {
      console.log('No previous benchmark data found');
      return;
    }
    
    console.log('\n📈 Performance Comparison vs Previous Run:\n');
    
    for (const [scenarioName, currentMetrics] of Object.entries(current.scenarios)) {
      const prevMetrics = previous.scenarios[scenarioName];
      
      if (!prevMetrics) {
        console.log(`${scenarioName}: NEW SCENARIO`);
        continue;
      }
      
      const opsChange = ((currentMetrics.ops_per_sec - prevMetrics.ops_per_sec) / prevMetrics.ops_per_sec) * 100;
      const latencyChange = ((currentMetrics.mean_ns - prevMetrics.mean_ns) / prevMetrics.mean_ns) * 100;
      
      const opsDirection = opsChange > 0 ? '⬆️' : opsChange < 0 ? '⬇️' : '➡️';
      const latencyDirection = latencyChange > 0 ? '⬆️' : latencyChange < 0 ? '⬇️' : '➡️';
      
      // Format latency appropriately
      const formatLatency = (ns: number) => {
        if (ns >= 1_000_000) return `${(ns / 1_000_000).toFixed(3)}ms`;
        if (ns >= 1_000) return `${(ns / 1_000).toFixed(1)}μs`;
        return `${ns.toFixed(0)}ns`;
      };
      
      console.log(`${scenarioName}:`);
      console.log(`  Throughput: ${opsDirection} ${opsChange.toFixed(2)}% (${currentMetrics.ops_per_sec.toFixed(0)} ops/sec)`);
      console.log(`  Latency: ${latencyDirection} ${latencyChange.toFixed(2)}% (${formatLatency(currentMetrics.mean_ns)})`);
      
      // Flag potential regressions
      if (opsChange < -5) {
        console.log(`  ⚠️  POTENTIAL REGRESSION: Throughput dropped by ${Math.abs(opsChange).toFixed(2)}%`);
      }
      if (latencyChange > 10) {
        console.log(`  ⚠️  POTENTIAL REGRESSION: Latency increased by ${latencyChange.toFixed(2)}%`);
      }
      console.log('');
    }
    
  } catch (error) {
    console.warn('Error comparing with history:', error);
  }
}