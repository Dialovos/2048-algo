#!/usr/bin/env node
// Headless check of the engine in 2048-algo.user.js:
//   1. compares its table-driven moves with a plain implementation on random boards,
//   2. plays full games with the real rules (new tile: 2 at 90%, 4 at 10%).
//
// Usage: node simulate.js [--games N] [--seed N] [--max-moves N] [--verify-only]
//   heuristic search (default): [--depth-bias N] [--max-depth N]
//   learned search:             --weights FILE [--search-depth N]

const fs = require('fs');
const { createEngine, networkLayout, NETWORK_LAYOUTS, DIR_NAMES } = require('./2048-algo.user.js');

const args = parseArgs(process.argv.slice(2));

function parseArgs(argv) {
  const out = { games: 3, seed: 1, maxMoves: Infinity, depthBias: 0, maxDepth: 20, weights: null, searchDepth: 3, verifyOnly: false };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === '--verify-only') out.verifyOnly = true;
    else if (flag === '--games') out.games = Number(argv[++i]);
    else if (flag === '--seed') out.seed = Number(argv[++i]);
    else if (flag === '--max-moves') out.maxMoves = Number(argv[++i]);
    else if (flag === '--depth-bias') out.depthBias = Number(argv[++i]);
    else if (flag === '--max-depth') out.maxDepth = Number(argv[++i]);
    else if (flag === '--weights') out.weights = argv[++i];
    else if (flag === '--search-depth') out.searchDepth = Number(argv[++i]);
    else throw new Error(`Unknown argument: ${flag}`);
  }
  return out;
}

// mulberry32: small seeded PRNG so runs are reproducible.
function makeRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Cell k of line i, where k = 0 is the edge the tiles move toward.
function cellAt(dir, i, k) {
  if (dir === 0) return [k, i]; // up
  if (dir === 1) return [i, 3 - k]; // right
  if (dir === 2) return [3 - k, i]; // down
  return [i, k]; // left
}

function referenceMove(grid, dir) {
  const next = grid.map((row) => row.slice());
  let gained = 0;
  for (let i = 0; i < 4; i++) {
    const tiles = [];
    for (let k = 0; k < 4; k++) {
      const [r, c] = cellAt(dir, i, k);
      if (grid[r][c]) tiles.push(grid[r][c]);
    }
    const merged = [];
    for (let t = 0; t < tiles.length; t++) {
      if (t + 1 < tiles.length && tiles[t] === tiles[t + 1]) {
        merged.push(tiles[t] * 2);
        gained += tiles[t] * 2;
        t++;
      } else {
        merged.push(tiles[t]);
      }
    }
    for (let k = 0; k < 4; k++) {
      const [r, c] = cellAt(dir, i, k);
      next[r][c] = merged[k] || 0;
    }
  }
  const changed = next.some((row, r) => row.some((v, c) => v !== grid[r][c]));
  return { grid: next, gained, changed };
}

function verifyMoves(engine, random, count) {
  for (let n = 0; n < count; n++) {
    // Ranks up to 16, so merges reach 131072, the largest tile a 4x4 board can hold.
    const grid = [0, 1, 2, 3].map(() => [0, 1, 2, 3].map(() => {
      const rank = random() < 0.35 ? 0 : 1 + Math.floor(random() * (random() < 0.8 ? 4 : 16));
      return rank ? 2 ** rank : 0;
    }));
    for (let dir = 0; dir < 4; dir++) {
      const expected = referenceMove(grid, dir);
      const actual = engine.move(grid, dir);
      const ok = (expected.changed
        ? actual !== null && JSON.stringify(actual) === JSON.stringify(expected.grid)
        : actual === null) && engine.reward(grid, dir) === expected.gained;
      if (!ok) {
        console.error('Move mismatch', DIR_NAMES[dir], JSON.stringify(grid));
        console.error('  expected', JSON.stringify(expected.changed ? expected.grid : null));
        console.error('  actual  ', JSON.stringify(actual));
        console.error('  reward expected', expected.gained, 'actual', engine.reward(grid, dir));
        process.exit(1);
      }
    }
  }
  console.log(`Moves and points match the reference on ${count} random boards x 4 directions.`);
}

// Late-game boards: the engine must keep choosing legal moves and score them above a loss (0).
function verifyBigTiles(engine) {
  const boards = {
    'merging two 32768s': [[32768, 32768, 16384, 8192], [256, 512, 1024, 4096], [128, 64, 32, 2048], [2, 4, 8, 16]],
    'holding 65536': [[65536, 32768, 16384, 8192], [256, 512, 1024, 4096], [128, 64, 32, 2048], [0, 2, 8, 16]],
    'building 131072': [[65536, 32768, 16384, 8192], [256, 512, 1024, 4096], [128, 64, 32, 2048], [0, 4, 8, 16]],
  };
  for (const [name, grid] of Object.entries(boards)) {
    const { dir, scores } = engine.bestMove(grid);
    if (dir < 0 || !referenceMove(grid, dir).changed || !(scores[dir] > 0)) {
      console.error(`Bad move on board "${name}": ${DIR_NAMES[dir]}, scores ${scores}`);
      process.exit(1);
    }
    console.log(`Board "${name}": plays ${DIR_NAMES[dir]} (heuristic ${Math.round(engine.evaluate(grid))}).`);
  }
}

function addTile(grid, random) {
  const empty = [];
  grid.forEach((row, r) => row.forEach((v, c) => {
    if (!v) empty.push([r, c]);
  }));
  const [r, c] = empty[Math.floor(random() * empty.length)];
  grid[r][c] = random() < 0.9 ? 2 : 4;
}

function playGame(engine, random, gameNo) {
  let grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  addTile(grid, random);
  addTile(grid, random);
  let score = 0, moves = 0, totalMs = 0, slowestMs = 0;

  while (moves < args.maxMoves) {
    const { dir, ms } = engine.bestMove(grid);
    if (dir < 0) break;
    const result = referenceMove(grid, dir);
    if (!result.changed) throw new Error(`Engine chose a no-op move: ${DIR_NAMES[dir]}`);
    grid = result.grid;
    score += result.gained;
    addTile(grid, random);
    moves++;
    totalMs += ms;
    slowestMs = Math.max(slowestMs, ms);
    if (moves % 1000 === 0) {
      const maxTile = Math.max(...grid.flat());
      console.log(`  game ${gameNo}: ${moves} moves, score ${score}, max tile ${maxTile}, avg ${(totalMs / moves).toFixed(1)} ms/move`);
    }
  }
  return { score, moves, maxTile: Math.max(...grid.flat()), avgMs: totalMs / Math.max(1, moves), slowestMs };
}

// One Float32Array per stage, read in chunks: Node can't read a 2.7 GB file in one call.
function readWeightsFile(file) {
  const size = fs.statSync(file).size;
  const layout = size % 4 ? null : networkLayout(size / 4);
  if (!layout) throw new Error(`${file} is not a weights file`);
  const { patterns, stages } = NETWORK_LAYOUTS[layout];
  const fd = fs.openSync(file, 'r');
  const parts = [];
  for (let s = 0; s < stages; s++) {
    const bytes = new Uint8Array((patterns << 24) * 4);
    for (let off = 0; off < bytes.length; off += 1 << 26) {
      fs.readSync(fd, bytes, off, Math.min(1 << 26, bytes.length - off), s * bytes.length + off);
    }
    parts.push(new Float32Array(bytes.buffer));
  }
  fs.closeSync(fd);
  return parts;
}

const random = makeRandom(args.seed);
const engine = createEngine({ depthBias: args.depthBias, maxDepth: args.maxDepth });
if (args.weights) engine.setWeights(readWeightsFile(args.weights));
verifyMoves(engine, random, 20000);
verifyBigTiles(engine);
if (args.verifyOnly) process.exit(0);
if (args.weights) {
  Object.assign(engine.options, { mode: 'learned', searchDepth: args.searchDepth });
  console.log(`Learned search, depth ${args.searchDepth}.`);
}

const results = [];
for (let g = 1; g <= args.games; g++) {
  const r = playGame(engine, random, g);
  results.push(r);
  console.log(`Game ${g}: score ${r.score}, max tile ${r.maxTile}, ${r.moves} moves, ` +
    `avg ${r.avgMs.toFixed(1)} ms/move, slowest ${r.slowestMs.toFixed(0)} ms`);
}

console.log('\nReached (share of games):');
for (const tile of [2048, 4096, 8192, 16384, 32768, 65536, 131072]) {
  const hits = results.filter((r) => r.maxTile >= tile).length;
  console.log(`  ${String(tile).padStart(5)}: ${hits}/${results.length}`);
}
const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
console.log(`Average score: ${Math.round(avgScore)}`);
