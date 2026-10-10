// The dashboard uses the same controller shape for CPU workers and CUDA.
const fs = require('fs');
const path = require('path');
const { spawn, execFile } = require('child_process');
const readline = require('readline');

const PYTHON = process.env.ALGO2048_PYTHON || path.join(__dirname, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const SCRIPT = path.join(__dirname, 'train-gpu.py');

function probeGpu() {
  return new Promise((resolve) => {
    execFile(PYTHON, [SCRIPT, '--probe'], { cwd: __dirname, timeout: 60000, maxBuffer: 1024 * 1024 }, (error, stdout) => {
      let result;
      try { result = JSON.parse(stdout.trim().split('\n').pop()); } catch (_) { /* dependency/runtime unavailable */ }
      resolve(!error && result && result.available ? result : {
        available: false,
        reason: result && result.message || 'GPU setup is unavailable. See the NVIDIA GPU setup in README.md.',
      });
    });
  });
}

function startGpuTraining(o, onLine) {
  const files = require('./train.js').networkFiles(o.weights);
  const parallelGames = o.gpuGames || 1024;
  if (!Number.isInteger(parallelGames) || parallelGames < 32 || parallelGames > 8192) throw new Error('GPU parallel games must be 32 to 8192.');
  if (!fs.existsSync(PYTHON)) throw new Error('GPU setup is missing. See README.md.');
  const args = [SCRIPT, '--control', '--weights', files.bin, '--games', String(parallelGames),
    '--restart', String(o.restart), '--optimism', String(o.optimistic),
    '--report-seconds', '5', '--save-minutes', String(o.saveMinutes)];
  if (Number.isFinite(o.alpha)) args.push('--alpha', String(o.alpha));
  if (o.noTc) args.push('--no-tc');
  // Only for a new network; the trainer refuses a starting network for an existing one.
  if (!fs.existsSync(files.bin)) {
    if (o.layout && o.layout !== 'standard') args.push('--layout', o.layout);
    if (o.source) args.push('--from', path.resolve(o.source));
  }
  // Hold a sleep/lid inhibitor for exactly the lifetime of Linux GPU training.
  const inhibit = process.platform === 'linux' && fs.existsSync('/usr/bin/systemd-inhibit');
  const executable = inhibit ? '/usr/bin/nohup' : PYTHON;
  const command = inhibit ? ['/usr/bin/systemd-inhibit', '--what=sleep:idle:handle-lid-switch',
    '--who=2048-training', '--why=GPU self-play training', PYTHON, ...args] : args;
  const child = spawn(executable, command, { cwd: __dirname, stdio: ['pipe', 'pipe', 'pipe'] });
  let stopped = false, status = 'starting', error = '', nextId = 0;
  let useTc = !o.noTc && (!fs.existsSync(files.bin) || fs.existsSync(files.tc));
  let alpha = Number.isFinite(o.alpha) ? o.alpha : useTc ? 1 : 0.1;
  let layout = 'standard';
  let summary = { elapsed: 0, games: 0, steps: 0, gamesPerSecond: 0, movesPerSecond: 0,
    newGames: 0, avg: 0, reach: {2048: 0, 4096: 0, 8192: 0, 16384: 0, 32768: 0}, restarts: {} };
  const pending = new Map();
  let closed;
  const finished = new Promise((resolve) => { closed = resolve; });
  const fail = (message) => { error = message; onLine(`GPU error: ${message}`); };
  child.on('error', (e) => fail(e.message));
  child.stdin.on('error', (e) => { if (!stopped) fail(e.message); });
  readline.createInterface({ input: child.stderr }).on('line', (line) => onLine(line));
  readline.createInterface({ input: child.stdout }).on('line', (line) => {
    let event;
    try { event = JSON.parse(line); } catch (_) { onLine(line); return; }
    if (event.type === 'log') onLine(event.message);
    else if (event.type === 'error') fail(event.message);
    else if (event.type === 'ready') {
      if (status !== 'stopping') status = 'training';
      useTc = event.tc; alpha = event.alpha; layout = event.layout;
      summary.games = event.totals.games; summary.steps = event.totals.steps;
    } else if (event.type === 'summary') summary = event.summary;
    else if (event.type === 'saved' && pending.has(event.id)) {
      pending.get(event.id).resolve(); pending.delete(event.id);
    }
  });
  child.on('close', (code, signal) => {
    stopped = true; status = 'stopped';
    if ((code !== 0 || signal) && !error) fail(`GPU worker exited (${signal || code}). Check the last saved checkpoint.`);
    for (const waiter of pending.values()) waiter.reject(new Error(error || 'GPU worker stopped before saving.'));
    pending.clear(); closed();
  });
  function commandRequest(command) {
    if (stopped) return Promise.reject(new Error(error || 'Not training.'));
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      child.stdin.write(JSON.stringify({ command, id }) + '\n');
    });
  }
  let stopping;
  return {
    files, backend: 'gpu', threads: 0, gpuGames: parallelGames, restart: o.restart,
    finished,
    startedAt: new Date().toISOString(), summary: () => summary,
    get useTc() { return useTc; }, get alpha() { return alpha; }, get layout() { return layout; },
    get stopped() { return stopped; }, get status() { return status; }, get error() { return error; },
    save: () => commandRequest('save'),
    stop() {
      if (!stopping) {
        status = 'stopping';
        stopping = (async () => {
          let failure;
          if (!stopped) {
            try { await commandRequest('stop'); } catch (e) { failure = e; }
          }
          await finished;
          if (failure || error) throw failure || new Error(error);
        })();
      }
      return stopping;
    },
  };
}

module.exports = { probeGpu, startGpuTraining };
