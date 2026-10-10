// Exercise the real JS engine as an independent oracle for the CUDA port.
const fs = require('fs');
const { createEngine, clearLateEntries, NETWORK_LAYOUTS } = require('../2048-algo.user.js');
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const layout = request.layout || 'standard';
const { patterns, stages } = NETWORK_LAYOUTS[layout];
const size = patterns << 24; // weights per stage

// Large layouts' arrays (1.6-2.7 GB of weights, twice that for TC) don't fit
// this machine's memory as dense arrays, so they use sparse stand-ins, one
// per stage: Float32Array rounding, each unset entry at its starting value.
function sparse(stage, initial) {
  const values = new Map();
  return new Proxy({ length: size, values }, {
    get(target, key) {
      if (typeof key !== 'string' || !/^\d+$/.test(key)) return target[key];
      const i = Number(key);
      return values.has(i) ? values.get(i) : initial(stage * size + i);
    },
    set(target, key, value) {
      values.set(Number(key), Math.fround(value));
      return true;
    },
  });
}
const stageArrays = (initial) => Array.from({ length: stages }, (_, s) => sparse(s, initial));
const lateEntry = (i) => i & (i >> 1) & (i >> 2) & 0x222222;
const dense = layout === 'standard';

let weights, tc = null, initial;
if (request.mode === 'boards') {
  initial = (i) => (i % 1024) / 128;
  weights = dense ? new Float32Array(size).map((_, i) => initial(i)) : stageArrays(initial);
  const engine = createEngine({ weights, cacheBits: 4 });
  const output = request.boards.map((grid) => ({ value: engine.value(grid),
    moves: [0, 1, 2, 3].map((dir) => ({ grid: engine.move(grid, dir), reward: engine.reward(grid, dir) })) }));
  process.stdout.write(JSON.stringify(output));
} else {
  // As the GPU starts a network: optimistic standard patterns, extra ones at 0.
  initial = (i) => (!dense && (i >> 24) % 8 >= 4) || lateEntry(i) ? 0 : 10000;
  if (dense) {
    weights = new Float32Array(size);
    clearLateEntries(weights.fill(10000));
    if (request.tc) tc = { e: new Float32Array(size), a: new Float32Array(size) };
  } else {
    weights = stageArrays(initial);
    if (request.tc) tc = { e: stageArrays(() => 0), a: stageArrays(() => 0) };
  }
  const engine = createEngine({ weights, tc, cacheBits: 4 });
  let rng = request.seed;
  const random = () => { rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5; return (rng >>> 0) / 4294967296; };
  for (let i = 0; i < (request.skipRandoms || 0); i++) random();
  const episodes = [];
  for (let i = 0; i < (request.episodes || 3); i++) episodes.push(engine.trainEpisode(request.startRows || null, 0.3, random));
  // Changed entries as [global index, weight, E, A].
  const changed = [];
  if (dense) {
    for (let i = 0; i < size; i++) {
      if (weights[i] !== initial(i) || (tc && tc.a[i] !== 0)) changed.push([i, weights[i], tc ? tc.e[i] : 0, tc ? tc.a[i] : 0]);
    }
  } else {
    for (let s = 0; s < stages; s++) {
      const touched = new Set([...weights[s].values.keys(), ...(tc ? tc.a[s].values.keys() : [])]);
      for (const i of [...touched].sort((x, y) => x - y)) {
        const w = weights[s][i], e = tc ? tc.e[s][i] : 0, a = tc ? tc.a[s][i] : 0;
        if (w !== initial(s * size + i) || a !== 0) changed.push([s * size + i, w, e, a]);
      }
    }
  }
  process.stdout.write(JSON.stringify({ episodes, changed }));
}
