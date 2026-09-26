#!/usr/bin/env node
// Trains the network behind the "Learned" search: self-play games with TD(0)
// learning on worker threads, all updating one shared set of weights. Run it
// directly (usage below) or from the dashboard (dashboard.js); both use
// startTraining(). New networks start optimistic and use TC learning: each
// weight adapts its own rate, so no rate schedule is needed.
//
// Usage: node train.js [options]
//   --weights FILE        weights to resume from and save to (default weights/network.bin)
//   --alpha N             step size (default 1 with TC learning; without it 0.1, dropping to 0.01, 0.001 as progress stalls)
//   --no-tc               new network without TC learning (fixed rate; uses a third of the memory)
//   --no-optimistic       new network starting every position at 0 instead of 320000
//   --threads N           worker threads (default: all CPU cores)
//   --restart P           share of games started from saved late-game boards (default 0.5)
//   --save-minutes N      checkpoint interval (default 10)
//   --report-seconds N    progress interval (default 60)
//   --hours N             stop after this long (default: run until Ctrl+C)
// Ctrl+C saves and exits; running again resumes from the saved file.

const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  createEngine, createSelfPlay, isNetworkSize, clearLateEntries, FEATURE_COUNT, NETWORK_WEIGHTS, OPTIMISTIC_VALUE,
} = require('./2048-algo.user.js');

const WINDOW = 10000; // new games in the rolling stats
const CHUNK = 1 << 26;
const REACH_TILES = [2048, 4096, 8192, 16384, 32768];

const DEFAULTS = {
  weights: 'weights/network.bin', alpha: NaN, noTc: false, optimistic: OPTIMISTIC_VALUE,
  threads: os.cpus().length, restart: 0.5, saveMinutes: 10, reportSeconds: 60,
};

// A network's files, named after its .bin file.
function networkFiles(bin) {
  const base = path.resolve(bin).replace(/\.bin$/, '');
  return { bin: base + '.bin', stats: base + '.json', log: base + '.log', tc: base + '.tc', history: base + '.history.jsonl' };
}

// Starts training on worker threads and returns a controller. Progress lines
// go to onLine and the network's .log file; each report is also appended to
// its .history.jsonl for charts.
function startTraining(options, onLine = () => {}) {
  const o = { ...DEFAULTS, ...options };
  const files = networkFiles(o.weights);
  fs.mkdirSync(path.dirname(files.bin), { recursive: true });
  const log = (line) => {
    onLine(line);
    fs.appendFileSync(files.log, `${new Date().toISOString()} ${line}\n`);
  };

  const exists = fs.existsSync(files.bin);
  const floats = exists ? fs.statSync(files.bin).size / 4 : NETWORK_WEIGHTS;
  if (!isNetworkSize(floats)) throw new Error(`${files.bin} is not a weights file (${NETWORK_WEIGHTS * 4} bytes; two-stage files from versions 1.4-1.5 aren't supported)`);
  const sab = new SharedArrayBuffer(floats * 4);
  const bytes = new Uint8Array(sab);
  // TC accumulators (E then A, per weight) live beside the weights. A network
  // saved without them keeps a fixed rate: fresh accumulators would restart
  // every trained weight at full step size.
  const useTc = !o.noTc && (!exists || fs.existsSync(files.tc));
  const tcSab = useTc ? new SharedArrayBuffer(floats * 8) : null;
  const tcBytes = useTc ? new Uint8Array(tcSab) : null;
  const alpha = Number.isFinite(o.alpha) ? o.alpha : useTc ? 1 : 0.1;

  const totals = { games: 0, steps: 0, seconds: 0 };
  if (exists) {
    readFile(files.bin, bytes);
    if (useTc) readFile(files.tc, tcBytes);
    if (fs.existsSync(files.stats)) Object.assign(totals, JSON.parse(fs.readFileSync(files.stats, 'utf8')));
    log(`Resumed ${path.basename(files.bin)} after ${totals.games.toLocaleString()} games.`);
  } else {
    clearLateEntries(new Float32Array(sab).fill(o.optimistic / FEATURE_COUNT));
    log(`Starting a new network at ${path.basename(files.bin)}, every position valued at ${o.optimistic.toLocaleString()} to start.`);
  }
  log(`${o.threads} threads, ${useTc ? `TC learning, step ${alpha}` : `fixed rate ${alpha}`}, restart share ${o.restart}.`);

  const startTime = Date.now();
  const startSeconds = totals.seconds;
  let speed = { gamesPerSecond: 0, movesPerSecond: 0 };
  let sample = { time: startTime, games: totals.games, steps: totals.steps };

  // Rolling stats: new games (score, max tile) and restarts (did they make the next tile).
  const fresh = new Ring(WINDOW, 2);
  const restarts = new Map();

  let stopped = false;
  const workers = [];
  for (let i = 0; i < o.threads; i++) {
    const w = new Worker(__filename, { workerData: { sab, tcSab, alpha, restart: o.restart } });
    w.on('message', (batch) => {
      for (let j = 0; j < batch.length; j += 4) {
        const [startRank, score, maxTile, steps] = [batch[j], batch[j + 1], batch[j + 2], batch[j + 3]];
        totals.games++;
        totals.steps += steps;
        if (startRank === 0) {
          fresh.push(score, maxTile);
        } else {
          if (!restarts.has(startRank)) restarts.set(startRank, new Ring(1000, 1));
          restarts.get(startRank).push(maxTile >= 2 ** (startRank + 1) ? 1 : 0);
        }
      }
    });
    w.on('error', (err) => {
      log(`A training thread failed: ${err.message}`);
      stop();
    });
    workers.push(w);
  }

  function summary() {
    const games = fresh.rows();
    const reach = {};
    for (const t of REACH_TILES) reach[t] = games.length ? games.filter((g) => g[1] >= t).length / games.length : 0;
    const restartRates = {};
    for (const [rank, ring] of [...restarts].sort((x, y) => x[0] - y[0])) {
      const hits = ring.rows();
      restartRates[2 ** rank] = hits.filter((h) => h[0]).length / hits.length;
    }
    return {
      time: new Date().toISOString(),
      elapsed: (Date.now() - startTime) / 1000,
      games: totals.games,
      steps: totals.steps,
      ...speed,
      newGames: games.length,
      avg: games.length ? games.reduce((s, g) => s + g[0], 0) / games.length : 0,
      reach,
      restarts: restartRates,
    };
  }

  function report() {
    const s = summary();
    const parts = [`${clock(s.elapsed * 1000)} ${s.games.toLocaleString()} games (${s.gamesPerSecond.toFixed(0)}/s, ${(s.movesPerSecond / 1e6).toFixed(2)}M moves/s)`];
    if (s.newGames) {
      parts.push(`new games: avg ${Math.round(s.avg).toLocaleString()}, ${REACH_TILES.map((t) => `${t} ${pct(s.reach[t])}`).join(' ')}`);
    }
    const r = Object.entries(s.restarts);
    if (r.length) parts.push(`restarts: ${r.map(([tile, rate]) => `${tile}->${tile * 2} ${pct(rate)}`).join(' ')}`);
    log(parts.join(' | '));
    fs.appendFileSync(files.history, JSON.stringify(s) + '\n');
  }

  function save() {
    writeFile(files.bin, bytes);
    if (useTc) writeFile(files.tc, tcBytes);
    totals.seconds = startSeconds + (Date.now() - startTime) / 1000;
    fs.writeFileSync(files.stats, JSON.stringify(totals, null, 2) + '\n');
  }

  const timers = [
    setInterval(() => {
      const now = Date.now(), secs = (now - sample.time) / 1000;
      speed = { gamesPerSecond: (totals.games - sample.games) / secs, movesPerSecond: (totals.steps - sample.steps) / secs };
      sample = { time: now, games: totals.games, steps: totals.steps };
    }, 5000),
    setInterval(report, o.reportSeconds * 1000),
    setInterval(() => {
      save();
      log('Checkpoint saved.');
    }, o.saveMinutes * 60000),
  ];

  // Saves and ends the threads. Safe to call more than once.
  function stop() {
    if (stopped) return;
    stopped = true;
    timers.forEach(clearInterval);
    workers.forEach((w) => w.terminate());
    report();
    save();
    log(`Saved ${path.basename(files.bin)}.`);
  }

  return {
    files, useTc, alpha, threads: o.threads, restart: o.restart, startedAt: new Date(startTime).toISOString(),
    summary,
    save() {
      save();
      log('Saved on request.');
    },
    stop,
    get stopped() {
      return stopped;
    },
  };
}

// The .bin networks in a folder, with what their side files say about them.
function listNetworks(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter((f) => f.endsWith('.bin')).map((f) => {
    const files = networkFiles(path.join(dir, f));
    const stat = fs.statSync(files.bin);
    let totals = { games: 0, seconds: 0 };
    try {
      totals = JSON.parse(fs.readFileSync(files.stats, 'utf8'));
    } catch (e) {
      // No stats yet (for example a file exported from the browser).
    }
    return {
      name: f.replace(/\.bin$/, ''),
      valid: isNetworkSize(stat.size / 4),
      tc: fs.existsSync(files.tc),
      bytes: stat.size + (fs.existsSync(files.tc) ? fs.statSync(files.tc).size : 0),
      games: totals.games,
      seconds: totals.seconds,
      updated: stat.mtime.toISOString(),
    };
  }).sort((a, b) => b.updated.localeCompare(a.updated));
}

// A network's report history, thinned evenly to at most maxPoints.
function readHistory(bin, maxPoints = 600) {
  const file = networkFiles(bin).history;
  if (!fs.existsSync(file)) return [];
  const points = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
  if (points.length <= maxPoints) return points;
  const step = points.length / maxPoints;
  return Array.from({ length: maxPoints }, (_, i) => points[Math.min(points.length - 1, Math.round(i * step))]);
}

function parseArgs(argv) {
  const out = { ...DEFAULTS, hours: Infinity };
  const names = {
    '--weights': 'weights', '--alpha': 'alpha', '--threads': 'threads', '--restart': 'restart',
    '--save-minutes': 'saveMinutes', '--report-seconds': 'reportSeconds', '--hours': 'hours',
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--no-tc' || argv[i] === '--no-optimistic') {
      if (argv[i] === '--no-tc') out.noTc = true;
      else out.optimistic = 0;
      continue;
    }
    const key = names[argv[i]];
    if (!key || i + 1 >= argv.length) throw new Error(`Unknown or incomplete argument: ${argv[i]}`);
    out[key] = key === 'weights' ? argv[++i] : Number(argv[++i]);
  }
  return out;
}

function cli() {
  const args = parseArgs(process.argv.slice(2));
  const trainer = startTraining(args, console.log);
  const finish = () => {
    trainer.stop();
    process.exit(0);
  };
  if (Number.isFinite(args.hours)) setTimeout(finish, args.hours * 3600000);
  process.on('SIGINT', finish);
}

function worker() {
  const { sab, tcSab, alpha, restart } = workerData;
  const weights = new Float32Array(sab);
  const tc = tcSab && { e: new Float32Array(tcSab, 0, weights.length), a: new Float32Array(tcSab, weights.length * 4, weights.length) };
  const engine = createEngine({ weights, tc, cacheBits: 4 });
  const selfPlay = createSelfPlay(engine, { alpha, restart });

  let batch = [], lastPost = Date.now();
  for (;;) {
    const r = selfPlay.play();
    batch.push(r.startRank, r.score, r.maxTile, r.steps);
    if (Date.now() - lastPost > 500) {
      parentPort.postMessage(batch);
      batch = [];
      lastPost = Date.now();
    }
  }
}

function readFile(name, bytes) {
  if (fs.statSync(name).size !== bytes.length) throw new Error(`${name} should be ${bytes.length} bytes`);
  const fd = fs.openSync(name, 'r');
  for (let off = 0; off < bytes.length;) off += fs.readSync(fd, bytes, off, Math.min(CHUNK, bytes.length - off), off);
  fs.closeSync(fd);
}

// Writes a temporary file first, so an interrupted save keeps the old file.
function writeFile(name, bytes) {
  const tmp = name + '.tmp';
  const fd = fs.openSync(tmp, 'w');
  for (let off = 0; off < bytes.length;) off += fs.writeSync(fd, bytes, off, Math.min(CHUNK, bytes.length - off));
  fs.closeSync(fd);
  fs.renameSync(tmp, name);
}

// Fixed-size window of the latest records, each `width` numbers.
class Ring {
  constructor(capacity, width) {
    this.data = new Float64Array(capacity * width);
    this.capacity = capacity;
    this.width = width;
    this.size = 0;
    this.next = 0;
  }

  push(...values) {
    this.data.set(values, this.next * this.width);
    this.next = (this.next + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  rows() {
    const out = [];
    for (let i = 0; i < this.size; i++) out.push(Array.from(this.data.subarray(i * this.width, (i + 1) * this.width)));
    return out;
  }
}

function pct(x) {
  return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}

function clock(ms) {
  const s = Math.floor(ms / 1000);
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map((n) => String(n).padStart(2, '0')).join(':');
}

module.exports = { startTraining, listNetworks, readHistory, networkFiles, DEFAULTS };

if (!isMainThread) worker();
else if (require.main === module) cli();
