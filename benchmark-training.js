#!/usr/bin/env node
// Short, in-memory comparisons. Source checkpoint and training totals are untouched.
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { performance } = require('perf_hooks');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { createEngine, createSelfPlay, clearLateEntries, NETWORK_WEIGHTS } = require('./2048-algo.user.js');

if (!isMainThread) {
  const weights = new Float32Array(workerData.weights);
  const tc = workerData.tc && { e: new Float32Array(workerData.tc, 0, weights.length), a: new Float32Array(workerData.tc, weights.length*4, weights.length) };
  let seed = 2048 + workerData.id;
  Math.random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 4294967296; };
  const trainer = createSelfPlay(createEngine({ weights, tc, cacheBits: 4 }), { alpha: tc ? 1 : 0.1, restart: 0.5 });
  parentPort.on('message', ms => {
    const start = performance.now(); let games = 0, moves = 0;
    do { const result = trainer.play(); games++; moves += result.steps; } while (performance.now()-start < ms);
    parentPort.postMessage({ games, moves });
  });
  parentPort.postMessage('ready');
} else {
  main().catch(error => { console.error(error); process.exit(1); });
}

async function main() {
  const args = process.argv.slice(2);
  const option = (flag, fallback) => args.includes(flag) ? args[args.indexOf(flag)+1] : fallback;
  const file = option('--weights', '');
  const seconds = Number(option('--seconds', '10'));
  const cpus = option('--cpu-threads', '8,24,32').split(',').filter(Boolean).map(Number);
  const gpus = option('--gpu-games', '256,1024,4096').split(',').filter(Boolean).map(Number);
  if (!Number.isFinite(seconds) || seconds <= 0 || cpus.some(n => !Number.isInteger(n) || n < 1 || n > 128) ||
      gpus.some(n => !Number.isInteger(n) || n < 32 || n > 8192)) throw new Error('Invalid benchmark duration or parallelism.');
  const existing = Boolean(file);
  if (existing && (!fs.existsSync(file) || fs.statSync(file).size !== NETWORK_WEIGHTS*4)) throw new Error('Expected a 256 MiB checkpoint.');
  const tcFile = file.replace(/\.bin$/, '.tc');
  const useTc = !existing || fs.existsSync(tcFile);
  console.log(JSON.stringify({ type: 'setup', checkpoint: existing ? path.basename(file) : 'fresh', tc: useTc,
    note: 'In-memory copies only; throughput is not a measure of playing strength.' }));
  const next = worker => new Promise((resolve, reject) => { worker.once('message', resolve); worker.once('error', reject); });
  for (const threads of cpus) {
    const weights = new SharedArrayBuffer(NETWORK_WEIGHTS*4);
    if (existing) new Uint8Array(weights).set(fs.readFileSync(file));
    else clearLateEntries(new Float32Array(weights).fill(10000));
    const tc = useTc ? new SharedArrayBuffer(NETWORK_WEIGHTS*8) : null;
    if (existing && useTc) new Uint8Array(tc).set(fs.readFileSync(tcFile));
    const workers = Array.from({length: threads}, (_,id) => new Worker(__filename, { workerData: {weights, tc, id} }));
    try {
      await Promise.all(workers.map(next));
      let responses = workers.map(next); workers.forEach(w => w.postMessage(2000)); await Promise.all(responses);
      responses = workers.map(next); const start = performance.now(); workers.forEach(w => w.postMessage(seconds*1000));
      const results = await Promise.all(responses); const elapsed = (performance.now()-start)/1000;
      const moves = results.reduce((sum,r) => sum+r.moves, 0);
      console.log(JSON.stringify({type: 'benchmark', backend: 'cpu', parallel: threads, tc: useTc,
        seconds: elapsed, games: results.reduce((sum,r) => sum+r.games, 0), moves, movesPerSecond: moves/elapsed}));
    } finally { await Promise.all(workers.map(w => w.terminate())); }
  }
  for (const games of gpus) {
    const python = process.env.ALGO2048_PYTHON || path.join(__dirname, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const args = ['train-gpu.py', '--benchmark', String(seconds), '--games', String(games),
      '--weights', file || path.join('.scratch', 'benchmark-never-saved.bin')];
    if (!file && fs.existsSync(args[args.length-1])) throw new Error('Remove the stale benchmark input first.');
    const child = spawn(python, args, { cwd: __dirname, stdio: ['ignore', 'inherit', 'inherit'] });
    await new Promise((resolve, reject) => { child.on('error', reject); child.on('close', code => code === 0 ? resolve() : reject(new Error(`GPU benchmark exited ${code}`))); });
  }
}
