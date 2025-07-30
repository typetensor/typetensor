import { Bench } from 'tinybench';

const bench = new Bench({ time: 1000 });

bench.add('simple test', () => {
  const x = Math.random();
  return x * 2;
});

await bench.run();

console.log('=== RAW TINYBENCH OUTPUT ===');
console.log('bench.tasks type:', typeof bench.tasks);
console.log('bench.tasks is Map:', bench.tasks instanceof Map);
console.log('bench.tasks size:', bench.tasks.size);

bench.tasks.forEach((task, name) => {
  console.log(`\nTask: ${name} (name type: ${typeof name})`);
  console.log('task.name:', task.name);
  
  const result = task.result;
  if (result) {
    console.log('Available properties:', Object.keys(result));
    console.log('hz:', result.hz);
    console.log('mean:', result.mean);
    console.log('sd:', result.sd);
    console.log('moe:', result.moe);
    console.log('p50:', result.p50);
    console.log('p95:', result.p95);
    console.log('p99:', result.p99);
    console.log('samples length:', result.samples?.length);
  } else {
    console.log('No result available');
  }
});