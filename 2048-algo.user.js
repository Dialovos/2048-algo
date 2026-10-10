// ==UserScript==
// @name         2048 Algo
// @namespace    2048-algo
// @version      1.8.0
// @description  Plays 2048.org with an expectimax search, using a hand-tuned heuristic or a trained n-tuple network. Adds a control panel to the game page.
// @match        https://2048.org/*
// @match        https://*.2048.org/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

/*
 * Runs three ways:
 *   - as a Tampermonkey/Violentmonkey userscript on 2048.org,
 *   - pasted into the DevTools console on the game page,
 *   - required from Node, where it only exports the engine (see simulate.js).
 *
 * Engine: expectimax search over a board stored as four 20-bit rows (one
 * 5-bit tile rank per cell, so every tile up to 131072 is exact), with move
 * and heuristic lookup tables for all 2^20 rows. Two evaluations:
 *   - heuristic: hand-tuned; search, heuristic and weights follow nneonneo/2048-ai,
 *   - learned: an n-tuple network trained by TD learning (see train.js).
 *
 * The heuristic search follows https://github.com/nneonneo/2048-ai, under
 * its license:
 *
 *   MIT License
 *
 *   Copyright (c) 2014-2019 Robert Xiao <nneonneo@gmail.com> and contributors.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */
(function () {
  'use strict';

  // Same numbering as the game's input manager.
  const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;
  const DIR_NAMES = ['up', 'right', 'down', 'left'];

  // Learned evaluation: 6-cell patterns (cells numbered 0-15, row by row),
  // each with 16^6 float32 weights. Changing these breaks saved weights. A
  // standard network uses the first four in one stage. Large and extra-large
  // networks (GPU training only) use all eight in each of 3 or 5 stages and
  // add a corner bonus; see "Learned evaluation" in createEngine. Files
  // differ in size, stage after stage.
  const NETWORK_TUPLES = [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6], [4, 5, 6, 8, 9, 10],
    [0, 1, 2, 3, 4, 8], [0, 1, 2, 4, 5, 8], [0, 1, 4, 5, 6, 7], [0, 1, 2, 5, 6, 7]];
  const NETWORK_LAYOUTS = {
    standard: { patterns: 4, stages: 1 },
    large: { patterns: 8, stages: 3 }, // stages up to 32768
    xlarge: { patterns: 8, stages: 5 }, // two more for 65536
  };
  const FEATURE_COUNT = 4 * 8; // standard: each pattern in 8 rotations/reflections
  const NETWORK_WEIGHTS = 4 << 24;
  const LARGE_NETWORK_WEIGHTS = (3 * 8) << 24;
  const XLARGE_NETWORK_WEIGHTS = (5 * 8) << 24;
  // New networks start every position at this many points (optimistic
  // initialization): positions training hasn't seen yet look good, so it explores them.
  const OPTIMISTIC_VALUE = 320000;

  // 'standard', 'large', 'xlarge', or null for a weights length.
  function networkLayout(length) {
    return Object.keys(NETWORK_LAYOUTS).find((name) => {
      const { patterns, stages } = NETWORK_LAYOUTS[name];
      return length === (stages * patterns) << 24;
    }) || null;
  }

  function isNetworkSize(length) {
    return networkLayout(length) !== null;
  }

  // Patterns holding 16384 or more only occur late in rare games, so their
  // weights train slowly. Starting them at 0 rather than optimistic keeps
  // untried ones from pulling the big tile toward whichever cells appear in
  // the most patterns (the middle), which wrecks the late game.
  function clearLateEntries(weights) {
    for (let t = 0; t < weights.length >> 24; t++) {
      const base = t << 24;
      for (let i = 0; i < 1 << 24; i++) {
        // Some cell of the pattern (4 bits each) holds rank 14 or 15: bits 1-3 of that nibble all set.
        if (i & (i >> 1) & (i >> 2) & 0x222222) weights[base + i] = 0;
      }
    }
  }

  // Optimistic, apart from the late-game entries.
  function newNetwork() {
    const weights = new Float32Array(NETWORK_WEIGHTS).fill(OPTIMISTIC_VALUE / FEATURE_COUNT);
    clearLateEntries(weights);
    return weights;
  }

  function createEngine(options) {
    const opts = Object.assign({
      mode: 'heuristic', // or 'learned', which needs weights
      depthBias: 0, maxDepth: 20, // heuristic search depth
      searchDepth: 3, // learned search: moves looked ahead, 1 = greedy
      cacheBits: 20,
      weights: null, // see setWeights
      tc: null, // { e, a }: Float32Arrays the size of weights, for TC learning (train.js)
    }, options);

    const LOST_PENALTY = 200000;
    const MONO_POWER = 4, MONO_WEIGHT = 47;
    const SUM_POWER = 3.5, SUM_WEIGHT = 11;
    const MERGES_WEIGHT = 700, EMPTY_WEIGHT = 270;
    // Branches less likely than this are scored by the heuristic instead of searched.
    const CPROB_THRESHOLD = 0.0001;
    // A lost board scores 0, so live boards must stay well above 0. The sum
    // penalty grows with rank^3.5 and can push big-tile boards below 0; when
    // the root scores under this margin, every leaf is lifted to restore it.
    const LOSS_MARGIN = 300000;

    const RANK_MASK = 31; // 5 bits per cell
    const ROW_COUNT = 1 << 20;

    const ROW_LEFT = new Uint32Array(ROW_COUNT);
    const ROW_RIGHT = new Uint32Array(ROW_COUNT);
    // Points scored by sliding a row; the same in both directions, since each
    // run of equal tiles merges the same number of pairs either way.
    const ROW_REWARD = new Uint32Array(ROW_COUNT);
    const HEUR = new Float64Array(ROW_COUNT);
    const EMPTY = new Uint8Array(ROW_COUNT);

    const sumTerm = [], monoTerm = [];
    for (let rank = 0; rank <= RANK_MASK; rank++) {
      sumTerm.push(Math.pow(rank, SUM_POWER));
      monoTerm.push(Math.pow(rank, MONO_POWER));
    }

    const reverseRow = (r) => ((r & 31) << 15) | (((r >> 5) & 31) << 10) | (((r >> 10) & 31) << 5) | (r >> 15);

    const line = new Int32Array(4);
    for (let row = 0; row < ROW_COUNT; row++) {
      line[0] = row & 31; line[1] = (row >> 5) & 31; line[2] = (row >> 10) & 31; line[3] = row >> 15;

      let sum = 0, empty = 0, merges = 0, prev = 0, counter = 0;
      for (let i = 0; i < 4; i++) {
        const rank = line[i];
        sum += sumTerm[rank];
        if (rank === 0) {
          empty++;
        } else {
          if (prev === rank) {
            counter++;
          } else if (counter > 0) {
            merges += 1 + counter;
            counter = 0;
          }
          prev = rank;
        }
      }
      if (counter > 0) merges += 1 + counter;

      let monoLeft = 0, monoRight = 0;
      for (let i = 1; i < 4; i++) {
        const a = monoTerm[line[i - 1]], b = monoTerm[line[i]];
        if (line[i - 1] > line[i]) monoLeft += a - b;
        else monoRight += b - a;
      }

      HEUR[row] = LOST_PENALTY + EMPTY_WEIGHT * empty + MERGES_WEIGHT * merges -
        MONO_WEIGHT * Math.min(monoLeft, monoRight) - SUM_WEIGHT * sum;
      EMPTY[row] = empty;

      // Slide toward index 0. Ranks above 17 never occur on a 4x4 board; the
      // cap only keeps those unused rows from overflowing their 5 bits.
      let reward = 0;
      for (let i = 0; i < 3; i++) {
        let j = i + 1;
        while (j < 4 && line[j] === 0) j++;
        if (j === 4) break;
        if (line[i] === 0) {
          line[i] = line[j];
          line[j] = 0;
          i--;
        } else if (line[i] === line[j]) {
          if (line[i] !== RANK_MASK) line[i]++;
          reward += 2 ** line[i];
          line[j] = 0;
        }
      }
      const result = line[0] | (line[1] << 5) | (line[2] << 10) | (line[3] << 15);
      ROW_LEFT[row] = result;
      ROW_RIGHT[reverseRow(row)] = reverseRow(result);
      ROW_REWARD[row] = reward;
    }

    // Column n of the board (s = 5 * n); applied to columns it transposes them back.
    function col(s, a, b, c, d) {
      return ((a >> s) & 31) | (((b >> s) & 31) << 5) | (((c >> s) & 31) << 10) | (((d >> s) & 31) << 15);
    }

    // doMove writes its result here to avoid allocating in the search.
    let o0 = 0, o1 = 0, o2 = 0, o3 = 0;

    function doMove(dir, a, b, c, d) {
      if (dir === LEFT) {
        o0 = ROW_LEFT[a]; o1 = ROW_LEFT[b]; o2 = ROW_LEFT[c]; o3 = ROW_LEFT[d];
      } else if (dir === RIGHT) {
        o0 = ROW_RIGHT[a]; o1 = ROW_RIGHT[b]; o2 = ROW_RIGHT[c]; o3 = ROW_RIGHT[d];
      } else {
        const T = dir === UP ? ROW_LEFT : ROW_RIGHT;
        const c0 = T[col(0, a, b, c, d)], c1 = T[col(5, a, b, c, d)];
        const c2 = T[col(10, a, b, c, d)], c3 = T[col(15, a, b, c, d)];
        o0 = col(0, c0, c1, c2, c3); o1 = col(5, c0, c1, c2, c3);
        o2 = col(10, c0, c1, c2, c3); o3 = col(15, c0, c1, c2, c3);
      }
      return o0 !== a || o1 !== b || o2 !== c || o3 !== d;
    }

    // Points scored by moving the board a, b, c, d in `dir`.
    function moveReward(dir, a, b, c, d) {
      if (dir === LEFT || dir === RIGHT) return ROW_REWARD[a] + ROW_REWARD[b] + ROW_REWARD[c] + ROW_REWARD[d];
      return ROW_REWARD[col(0, a, b, c, d)] + ROW_REWARD[col(5, a, b, c, d)] +
        ROW_REWARD[col(10, a, b, c, d)] + ROW_REWARD[col(15, a, b, c, d)];
    }

    function heuristic(a, b, c, d) {
      return HEUR[a] + HEUR[b] + HEUR[c] + HEUR[d] +
        HEUR[col(0, a, b, c, d)] + HEUR[col(5, a, b, c, d)] +
        HEUR[col(10, a, b, c, d)] + HEUR[col(15, a, b, c, d)];
    }

    function distinctTiles(a, b, c, d) {
      let bits = 0;
      for (const r of [a, b, c, d]) {
        for (let s = 0; s < 20; s += 5) bits |= 1 << ((r >> s) & 31);
      }
      bits >>>= 1; // drop empty cells
      let n = 0;
      while (bits) {
        bits &= bits - 1;
        n++;
      }
      return n;
    }

    // Direct-mapped transposition table; bumping `gen` clears it between searches.
    // Each key packs two 20-bit rows into one exact double.
    const TT_SHIFT = 32 - opts.cacheBits;
    const ttKey1 = new Float64Array(1 << opts.cacheBits);
    const ttKey2 = new Float64Array(1 << opts.cacheBits);
    const ttGen = new Uint32Array(1 << opts.cacheBits);
    const ttDepth = new Uint8Array(1 << opts.cacheBits);
    const ttScore = new Float64Array(1 << opts.cacheBits);
    let gen = 0;

    let depthLimit = 0, evals = 0, leafOffset = 0;

    // Average over every empty cell receiving a 2 (90%) or a 4 (10%).
    function chanceNode(a, b, c, d, cprob, depth) {
      if (cprob < CPROB_THRESHOLD || depth >= depthLimit) {
        evals++;
        return heuristic(a, b, c, d) + leafOffset;
      }

      const key1 = a * ROW_COUNT + b, key2 = c * ROW_COUNT + d;
      const slot = Math.imul(a ^ Math.imul(b, 0x85ebca6b) ^ Math.imul(c, 0xc2b2ae35) ^ Math.imul(d, 0x27d4eb2f), 0x9e3779b1) >>> TT_SHIFT;
      if (ttGen[slot] === gen && ttKey1[slot] === key1 && ttKey2[slot] === key2 && ttDepth[slot] <= depth) {
        return ttScore[slot];
      }

      const empty = EMPTY[a] + EMPTY[b] + EMPTY[c] + EMPTY[d];
      const p2 = cprob * 0.9 / empty, p4 = cprob * 0.1 / empty;
      let total = 0;
      for (let s = 0; s < 20; s += 5) {
        if (((a >> s) & 31) === 0) {
          total += 0.9 * maxNode(a | (1 << s), b, c, d, p2, depth) + 0.1 * maxNode(a | (2 << s), b, c, d, p4, depth);
        }
        if (((b >> s) & 31) === 0) {
          total += 0.9 * maxNode(a, b | (1 << s), c, d, p2, depth) + 0.1 * maxNode(a, b | (2 << s), c, d, p4, depth);
        }
        if (((c >> s) & 31) === 0) {
          total += 0.9 * maxNode(a, b, c | (1 << s), d, p2, depth) + 0.1 * maxNode(a, b, c | (2 << s), d, p4, depth);
        }
        if (((d >> s) & 31) === 0) {
          total += 0.9 * maxNode(a, b, c, d | (1 << s), p2, depth) + 0.1 * maxNode(a, b, c, d | (2 << s), p4, depth);
        }
      }
      const score = total / empty;

      ttGen[slot] = gen; ttKey1[slot] = key1; ttKey2[slot] = key2; ttDepth[slot] = depth; ttScore[slot] = score;
      return score;
    }

    // Best move for the player; a board with no moves scores 0 (lost).
    function maxNode(a, b, c, d, cprob, depth) {
      let best = 0;
      depth++;
      for (let dir = 0; dir < 4; dir++) {
        if (doMove(dir, a, b, c, d)) {
          const v = chanceNode(o0, o1, o2, o3, cprob, depth);
          if (v > best) best = v;
        }
      }
      return best;
    }

    // ---- Learned evaluation ------------------------------------------------
    // The network estimates the points still to come from an afterstate (the
    // board after a move, before the new tile). Each pattern is read in all 8
    // rotations and reflections: 32 weight lookups per board in a standard
    // network, 64 in a large one. Ranks are capped at 15 inside the network,
    // so tiles from 32768 up look alike to it.
    //
    // Large networks keep a separate set of weights for each stage of the
    // game: before 16384, then two stages for each big-tile level: holding
    // 16384 (and in extra-large networks, 32768) with no other tile of 8192
    // or more, and with one (building the next big tile). A large network has
    // one level, so it treats 32768 boards like 16384 ones; an extra-large
    // network has two. Both add a corner bonus to every value (potential below).

    let FEATURES = FEATURE_COUNT, levels = 0;
    const featureCells = new Int32Array(NETWORK_TUPLES.length * 8 * 6);
    {
      let f = 0;
      for (const tuple of NETWORK_TUPLES) {
        for (let rot = 0; rot < 4; rot++) {
          for (let flip = 0; flip < 2; flip++, f++) {
            tuple.forEach((cell, j) => {
              let r = cell >> 2, c = cell & 3;
              if (flip) c = 3 - c;
              for (let k = 0; k < rot; k++) [r, c] = [c, 3 - r];
              featureCells[f * 6 + j] = r * 4 + c;
            });
          }
        }
      }
    }
    const NET_RANK = new Int32Array(32).map((_, rank) => Math.min(rank, 15));
    const cells = new Int32Array(16);
    const featureIndex = new Int32Array(NETWORK_TUPLES.length * 8);
    // Weights and TC accumulators, one array per stage; W, TCE and TCA are the
    // current board's stage (set by indexFeatures).
    let stageW = null, stageE = null, stageA = null, W = null, TCE = null, TCA = null;

    // Fills featureIndex with each feature's weight index for this board.
    function indexFeatures(a, b, c, d) {
      for (let i = 0, s = 0; i < 4; i++, s += 5) {
        cells[i] = NET_RANK[(a >> s) & 31];
        cells[4 + i] = NET_RANK[(b >> s) & 31];
        cells[8 + i] = NET_RANK[(c >> s) & 31];
        cells[12 + i] = NET_RANK[(d >> s) & 31];
      }
      let stage = 0;
      if (levels) {
        let top = 0, big = 0;
        for (let i = 0; i < 16; i++) {
          if (cells[i] > top) top = cells[i];
          if (cells[i] >= 13) big++;
        }
        if (top >= 14) stage = 1 + 2 * (Math.min(top, 13 + levels) - 14) + (big >= 2 ? 1 : 0);
      }
      W = stageW[stage];
      if (stageE) {
        TCE = stageE[stage];
        TCA = stageA[stage];
      }
      for (let f = 0, p = 0; f < FEATURES; f++, p += 6) {
        featureIndex[f] = ((f >> 3) << 24) | cells[featureCells[p]] | (cells[featureCells[p + 1]] << 4) |
          (cells[featureCells[p + 2]] << 8) | (cells[featureCells[p + 3]] << 12) |
          (cells[featureCells[p + 4]] << 16) | (cells[featureCells[p + 5]] << 20);
      }
    }

    function indexedValue() {
      let v = 0;
      for (let f = 0; f < FEATURES; f++) v += W[featureIndex[f]];
      return v;
    }

    // Large layouts: the biggest tile's value when it sits in a corner, plus
    // half the bigger of that corner's two edge neighbours; 0 otherwise.
    // Training learns the value minus this bonus (potential-based reward
    // shaping), so the best moves by final score don't change, but a move that
    // pulls the big tile out of its corner costs points at once instead of
    // only many moves later, and search keeps the bonus in rare positions the
    // weights haven't learned.
    const CORNER_CELLS = [0, 1, 4, 3, 2, 7, 12, 13, 8, 15, 14, 11]; // corner, then its edge neighbours
    function rankAt(a, b, c, d, cell) {
      return ((cell < 4 ? a : cell < 8 ? b : cell < 12 ? c : d) >> (5 * (cell & 3))) & 31;
    }
    function potential(a, b, c, d) {
      if (!levels) return 0;
      let top = 0;
      for (let s = 0; s < 20; s += 5) top = Math.max(top, (a >> s) & 31, (b >> s) & 31, (c >> s) & 31, (d >> s) & 31);
      let bonus = 0;
      for (let k = 0; k < 12; k += 3) {
        if (rankAt(a, b, c, d, CORNER_CELLS[k]) !== top) continue;
        const next = Math.max(rankAt(a, b, c, d, CORNER_CELLS[k + 1]), rankAt(a, b, c, d, CORNER_CELLS[k + 2]));
        bonus = Math.max(bonus, 2 ** top + (next ? 2 ** next / 2 : 0));
      }
      return bonus;
    }

    function netValue(a, b, c, d) {
      indexFeatures(a, b, c, d);
      return indexedValue() + potential(a, b, c, d);
    }

    // For choosing moves: a weight still at its optimistic starting value has
    // never been trained, so it counts as 0 (unknown) instead of flattering
    // whatever rare position the search reaches. Training keeps it optimistic.
    const UNTRAINED = Math.fround(OPTIMISTIC_VALUE / FEATURE_COUNT);
    function playValue(a, b, c, d) {
      indexFeatures(a, b, c, d);
      let v = 0;
      for (let f = 0; f < FEATURES; f++) {
        const w = W[featureIndex[f]];
        if (w !== UNTRAINED) v += w;
      }
      return v + potential(a, b, c, d);
    }

    // One learning step: move this afterstate's value toward `target`; returns
    // the new value. With TC accumulators, each weight scales the step by its
    // own rate |E| / A (summed errors over summed error sizes), which shrinks
    // as its errors stop agreeing in sign (temporal coherence learning).
    function netLearn(a, b, c, d, target, alpha) {
      indexFeatures(a, b, c, d);
      const bonus = potential(a, b, c, d);
      const error = target - (indexedValue() + bonus);
      const step = alpha * error / FEATURES;
      if (TCE) {
        const size = Math.abs(error);
        for (let f = 0; f < FEATURES; f++) {
          const i = featureIndex[f], acc = TCA[i];
          W[i] += acc === 0 ? step : step * Math.abs(TCE[i]) / acc;
          TCE[i] += error;
          TCA[i] = acc + size;
        }
      } else {
        for (let f = 0; f < FEATURES; f++) W[featureIndex[f]] += step;
      }
      return indexedValue() + bonus;
    }

    // Like chanceNode, but leaves are network values and `depth` counts down.
    function netChance(a, b, c, d, cprob, depth) {
      if (depth <= 0 || cprob < CPROB_THRESHOLD) {
        evals++;
        return playValue(a, b, c, d);
      }

      const key1 = a * ROW_COUNT + b, key2 = c * ROW_COUNT + d;
      const slot = Math.imul(a ^ Math.imul(b, 0x85ebca6b) ^ Math.imul(c, 0xc2b2ae35) ^ Math.imul(d, 0x27d4eb2f), 0x9e3779b1) >>> TT_SHIFT;
      if (ttGen[slot] === gen && ttKey1[slot] === key1 && ttKey2[slot] === key2 && ttDepth[slot] >= depth) {
        return ttScore[slot];
      }

      const empty = EMPTY[a] + EMPTY[b] + EMPTY[c] + EMPTY[d];
      const p2 = cprob * 0.9 / empty, p4 = cprob * 0.1 / empty;
      let total = 0;
      for (let s = 0; s < 20; s += 5) {
        if (((a >> s) & 31) === 0) {
          total += 0.9 * netMax(a | (1 << s), b, c, d, p2, depth) + 0.1 * netMax(a | (2 << s), b, c, d, p4, depth);
        }
        if (((b >> s) & 31) === 0) {
          total += 0.9 * netMax(a, b | (1 << s), c, d, p2, depth) + 0.1 * netMax(a, b | (2 << s), c, d, p4, depth);
        }
        if (((c >> s) & 31) === 0) {
          total += 0.9 * netMax(a, b, c | (1 << s), d, p2, depth) + 0.1 * netMax(a, b, c | (2 << s), d, p4, depth);
        }
        if (((d >> s) & 31) === 0) {
          total += 0.9 * netMax(a, b, c, d | (1 << s), p2, depth) + 0.1 * netMax(a, b, c, d | (2 << s), p4, depth);
        }
      }
      const score = total / empty;

      ttGen[slot] = gen; ttKey1[slot] = key1; ttKey2[slot] = key2; ttDepth[slot] = depth; ttScore[slot] = score;
      return score;
    }

    // Points from the best move plus the value after it; a board with no moves has nothing left to score.
    function netMax(a, b, c, d, cprob, depth) {
      let best = -Infinity;
      for (let dir = 0; dir < 4; dir++) {
        if (doMove(dir, a, b, c, d)) {
          const v = moveReward(dir, a, b, c, d) + netChance(o0, o1, o2, o3, cprob, depth - 1);
          if (v > best) best = v;
        }
      }
      return best === -Infinity ? 0 : best;
    }

    // Writes a, b, c, d plus one random new tile (2 at 90%, 4 at 10%) to o0..o3.
    function spawn(a, b, c, d, random) {
      let k = Math.floor(random() * (EMPTY[a] + EMPTY[b] + EMPTY[c] + EMPTY[d]));
      const tile = random() < 0.9 ? 1 : 2;
      const rows = [a, b, c, d];
      for (let r = 0; r < 4; r++) {
        for (let s = 0; s < 20; s += 5) {
          if (((rows[r] >> s) & 31) === 0 && k-- === 0) rows[r] |= tile << s;
        }
      }
      [o0, o1, o2, o3] = rows;
    }

    function maxRank(a, b, c, d) {
      let m = 0;
      for (const r of [a, b, c, d]) {
        for (let s = 0; s < 20; s += 5) m = Math.max(m, (r >> s) & 31);
      }
      return m;
    }

    let episodeAfter = new Int32Array(4 * 8192), episodeReward = new Float64Array(8192);

    // Training (Node): plays one game greedily on the network, then applies
    // TD(0) backward over its afterstates. Starts from `rows` (a board after a
    // tile spawned) or a new game. onNewTile(rank, rows) sees the board each
    // time a new largest tile appears, for restarting later episodes there.
    function trainEpisode(rows, alpha, random, onNewTile) {
      let a, b, c, d;
      if (rows) {
        [a, b, c, d] = rows;
      } else {
        spawn(0, 0, 0, 0, random);
        spawn(o0, o1, o2, o3, random);
        a = o0; b = o1; c = o2; d = o3;
      }
      let top = maxRank(a, b, c, d), score = 0, steps = 0;

      for (;;) {
        let best = -Infinity, reward = 0, n0 = 0, n1 = 0, n2 = 0, n3 = 0;
        for (let dir = 0; dir < 4; dir++) {
          if (!doMove(dir, a, b, c, d)) continue;
          const r = moveReward(dir, a, b, c, d);
          const v = r + netValue(o0, o1, o2, o3);
          if (v > best) {
            best = v; reward = r; n0 = o0; n1 = o1; n2 = o2; n3 = o3;
          }
        }
        if (best === -Infinity) break;

        if (steps === episodeReward.length) {
          const after = new Int32Array(episodeAfter.length * 2);
          after.set(episodeAfter);
          episodeAfter = after;
          const rewards = new Float64Array(episodeReward.length * 2);
          rewards.set(episodeReward);
          episodeReward = rewards;
        }
        episodeAfter[4 * steps] = n0; episodeAfter[4 * steps + 1] = n1;
        episodeAfter[4 * steps + 2] = n2; episodeAfter[4 * steps + 3] = n3;
        episodeReward[steps] = reward;
        steps++;
        score += reward;

        spawn(n0, n1, n2, n3, random);
        a = o0; b = o1; c = o2; d = o3;
        if (reward >= 2 ** (top + 1)) {
          const m = maxRank(a, b, c, d);
          if (m > top) {
            top = m;
            if (onNewTile) onNewTile(top, [a, b, c, d]);
          }
        }
      }

      let target = 0; // the last afterstate led to a lost board
      for (let t = steps - 1; t >= 0; t--) {
        const i = 4 * t;
        target = episodeReward[t] + netLearn(episodeAfter[i], episodeAfter[i + 1], episodeAfter[i + 2], episodeAfter[i + 3], target, alpha);
      }
      return { score, steps, maxTile: 2 ** top };
    }

    // grid[row][col] holds tile values, 0 for empty.
    function toRows(grid) {
      const rows = [0, 0, 0, 0];
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          const v = grid[r][c];
          if (v) rows[r] |= Math.min(RANK_MASK, Math.round(Math.log2(v))) << (5 * c);
        }
      }
      return rows;
    }

    function toGrid(rows) {
      return rows.map((r) => [0, 5, 10, 15].map((s) => {
        const rank = (r >> s) & 31;
        return rank ? 2 ** rank : 0;
      }));
    }

    const now = typeof performance !== 'undefined' ? () => performance.now() : () => Date.now();

    // Returns { dir: -1 } when no move changes the board (game over).
    function bestMove(grid) {
      const [a, b, c, d] = toRows(grid);
      const learned = opts.mode === 'learned';
      if (learned && !W) throw new Error('Learned mode needs weights');
      const start = now();
      gen++;
      evals = 0;
      depthLimit = learned ? opts.searchDepth
        : Math.min(opts.maxDepth, Math.max(3, distinctTiles(a, b, c, d) - 2 + opts.depthBias));
      leafOffset = learned ? 0 : Math.max(0, LOSS_MARGIN - heuristic(a, b, c, d));

      const scores = [0, 0, 0, 0];
      let dir = -1;
      for (let m = 0; m < 4; m++) {
        if (!doMove(m, a, b, c, d)) continue;
        scores[m] = learned
          ? moveReward(m, a, b, c, d) + netChance(o0, o1, o2, o3, 1, opts.searchDepth - 1)
          : chanceNode(o0, o1, o2, o3, 1, 0) + 1e-6;
        if (dir < 0 || scores[m] > scores[dir]) dir = m;
      }
      return { dir, scores, depthLimit, evals, ms: now() - start };
    }

    // The board after a move, or null if the move changes nothing.
    function move(grid, dir) {
      const [a, b, c, d] = toRows(grid);
      return doMove(dir, a, b, c, d) ? toGrid([o0, o1, o2, o3]) : null;
    }

    // Points a move scores (0 if it merges nothing).
    function reward(grid, dir) {
      const [a, b, c, d] = toRows(grid);
      return moveReward(dir, a, b, c, d);
    }

    // Heuristic score of a board, before any leaf lift.
    function evaluate(grid) {
      const [a, b, c, d] = toRows(grid);
      return heuristic(a, b, c, d);
    }

    // Afterstate value from the network (points still to come, with any corner bonus).
    function value(grid) {
      const [a, b, c, d] = toRows(grid);
      return netValue(a, b, c, d);
    }

    // weights (and tc.e, tc.a) are one Float32Array for the whole network or
    // one per stage; browsers and Node can't hold a 2.7 GB network in one array.
    function setWeights(weights, tc) {
      const parts = Array.isArray(weights) ? weights : [weights];
      const layout = networkLayout(parts.reduce((n, part) => n + part.length, 0));
      if (!layout) throw new Error('Wrong weights size');
      const { patterns, stages } = NETWORK_LAYOUTS[layout], size = patterns << 24;
      const split = (list) => {
        if (list.length === stages && list.every((part) => part.length === size)) return list;
        if (list.length !== 1) throw new Error('Wrong weights size');
        return Array.from({ length: stages }, (_, s) => list[0].subarray(s * size, (s + 1) * size));
      };
      levels = (stages - 1) / 2;
      FEATURES = patterns * 8;
      stageW = split(parts);
      stageE = tc ? split(Array.isArray(tc.e) ? tc.e : [tc.e]) : null;
      stageA = tc ? split(Array.isArray(tc.a) ? tc.a : [tc.a]) : null;
      W = stageW[0];
      TCE = TCA = null;
    }
    if (opts.weights) setWeights(opts.weights, opts.tc);

    return { options: opts, bestMove, move, reward, evaluate, value, setWeights, trainEpisode };
  }

  // Self-play for training (train.js and the in-browser trainer). A share of
  // games (`restart`) starts from boards where an earlier game first made
  // 8192 or more, so late-game positions get practice full games rarely reach.
  function createSelfPlay(engine, options) {
    const POOL_MIN_RANK = 13, POOL_SIZE = 1000;
    const pools = new Map(); // rank -> boards (rows) where that tile first appeared
    const keep = (rank, rows) => {
      if (rank < POOL_MIN_RANK) return;
      if (!pools.has(rank)) pools.set(rank, []);
      const pool = pools.get(rank);
      if (pool.length < POOL_SIZE) pool.push(rows);
      else pool[Math.floor(Math.random() * POOL_SIZE)] = rows;
    };
    const selfPlay = {
      alpha: options.alpha,
      restart: options.restart,
      // Returns { startRank (0 for a new game), score, maxTile, steps }.
      play() {
        let start = null, startRank = 0;
        if (pools.size && Math.random() < selfPlay.restart) {
          const ranks = [...pools.keys()];
          startRank = ranks[Math.floor(Math.random() * ranks.length)];
          const pool = pools.get(startRank);
          start = pool[Math.floor(Math.random() * pool.length)];
        }
        const r = engine.trainEpisode(start, selfPlay.alpha, Math.random, keep);
        return { startRank, score: r.score, maxTile: r.maxTile, steps: r.steps };
      },
    };
    return selfPlay;
  }

  // A weights file or stored blob as the engine takes it: one Float32Array for
  // a standard network, one per stage for larger ones (a browser can't hold
  // 2.7 GB in one array). Reads one stage at a time.
  async function readWeights(blob) {
    const layout = blob.size % 4 ? null : networkLayout(blob.size / 4);
    if (!layout) throw new Error('Not a weights file');
    const { patterns, stages } = NETWORK_LAYOUTS[layout];
    if (stages === 1) return new Float32Array(await blob.arrayBuffer());
    const bytes = (patterns << 24) * 4, parts = [];
    for (let s = 0; s < stages; s++) parts.push(new Float32Array(await blob.slice(s * bytes, (s + 1) * bytes).arrayBuffer()));
    return parts;
  }

  // One IndexedDB request against this site's 'algo2048' store (page or worker).
  function idb(mode, request) {
    return new Promise((resolve, reject) => {
      const open = indexedDB.open('algo2048', 1);
      open.onupgradeneeded = () => open.result.createObjectStore('files');
      open.onerror = () => reject(open.error);
      open.onsuccess = () => {
        const db = open.result;
        const tx = db.transaction('files', mode);
        const req = request(tx.objectStore('files'));
        tx.oncomplete = () => {
          db.close();
          resolve(req.result);
        };
        tx.onerror = tx.onabort = () => {
          db.close();
          reject(tx.error);
        };
      };
    });
  }

  // Body of the in-browser training worker. It owns the weights while it
  // trains, saves them to IndexedDB every few minutes and when stopped, then
  // hands them back to the page.
  function trainingWorker() {
    const SAVE_MS = 5 * 60000, REPORT_MS = 2000, WINDOW = 2000;
    let weights = null, selfPlay = null, totals = null, running = false, saving = null;
    let lastSave = 0, lastReport = 0, reportSteps = 0, recentNext = 0;
    const recent = []; // ring of [score, maxTile] for the latest new games

    self.onmessage = async (e) => {
      const msg = e.data;
      if (msg.type === 'start') {
        weights = new Float32Array(msg.buffer);
        totals = msg.totals;
        selfPlay = createSelfPlay(createEngine({ weights, cacheBits: 4 }), { alpha: msg.alpha, restart: 0.5 });
        running = true;
        lastSave = lastReport = performance.now();
        setTimeout(trainChunk, 0);
      } else if (msg.type === 'alpha') {
        if (selfPlay) selfPlay.alpha = msg.alpha;
      } else if (msg.type === 'stop') {
        running = false;
        let error = null;
        try {
          await save();
        } catch (err) {
          error = String(err);
        }
        self.postMessage({ type: 'stopped', buffer: weights.buffer, totals, error }, [weights.buffer]);
      }
    };

    // Trains for ~100 ms, then yields so 'stop' and 'alpha' messages get
    // through. Worker timers aren't throttled like a background page's.
    function trainChunk() {
      if (!running) return;
      const start = performance.now();
      while (performance.now() - start < 100) {
        const r = selfPlay.play();
        totals.games++;
        totals.steps += r.steps;
        reportSteps += r.steps;
        if (r.startRank === 0) {
          recent[recentNext] = [r.score, r.maxTile];
          recentNext = (recentNext + 1) % WINDOW;
        }
      }
      const now = performance.now();
      totals.seconds += (now - start) / 1000;
      if (now - lastReport > REPORT_MS) {
        const reach = {};
        for (const tile of [2048, 4096, 8192, 16384, 32768]) reach[tile] = recent.filter((g) => g[1] >= tile).length / (recent.length || 1);
        const avg = recent.reduce((s, g) => s + g[0], 0) / (recent.length || 1);
        self.postMessage({ type: 'progress', totals, avg, reach, movesPerSecond: reportSteps / ((now - lastReport) / 1000) });
        lastReport = now;
        reportSteps = 0;
      }
      if (now - lastSave > SAVE_MS && !saving) {
        lastSave = now;
        save().catch((err) => self.postMessage({ type: 'error', error: String(err) }));
      }
      setTimeout(trainChunk, 0);
    }

    async function save() {
      if (saving) await saving;
      const snapshot = new Blob([weights]); // copies, so training can go on during the write
      saving = idb('readwrite', (store) => {
        store.put(snapshot, 'weights');
        return store.put({ ...totals }, 'stats');
      });
      try {
        await saving;
        self.postMessage({ type: 'saved', totals });
      } finally {
        saving = null;
      }
    }
  }

  // Worker source: the engine and trainer functions above, with the
  // constants they use (functions keep their source text in toString()).
  function trainingWorkerSource() {
    return [
      "'use strict';",
      `const UP = ${UP}, RIGHT = ${RIGHT}, DOWN = ${DOWN}, LEFT = ${LEFT};`,
      `const NETWORK_TUPLES = ${JSON.stringify(NETWORK_TUPLES)};`,
      `const FEATURE_COUNT = ${FEATURE_COUNT};`,
      `const NETWORK_LAYOUTS = ${JSON.stringify(NETWORK_LAYOUTS)};`,
      `const NETWORK_WEIGHTS = ${NETWORK_WEIGHTS};`,
      `const OPTIMISTIC_VALUE = ${OPTIMISTIC_VALUE};`,
      networkLayout.toString(),
      isNetworkSize.toString(),
      createEngine.toString(),
      createSelfPlay.toString(),
      idb.toString(),
      trainingWorker.toString(),
      'trainingWorker();',
    ].join('\n');
  }

  function installDriver() {
    const tileContainer = document.querySelector('.tile-container');
    const messageEl = document.querySelector('.game-message');
    const keepPlayingButton = document.querySelector('.keep-playing-button');
    if (!tileContainer || !messageEl || !keepPlayingButton) return; // not a game page

    if (window.algo2048) window.algo2048.destroy(); // pasted into the console twice

    const KEY_CODES = [38, 39, 40, 37];
    const KEY_NAMES = ['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'];
    const SPEEDS = { instant: 0, fast: 40, watch: 200 };
    const STRENGTHS = {
      best: { mode: 'heuristic', depthBias: 0, maxDepth: 20 },
      quick: { mode: 'heuristic', depthBias: -1, maxDepth: 5 },
      learned: { mode: 'learned', searchDepth: 4 },
      'learned-deep': { mode: 'learned', searchDepth: 5 },
    };

    // Built on first use: the lookup tables take a moment and about 50 MB.
    let engine = null;
    let strength = 'best';
    const getEngine = () => engine || (engine = createEngine(STRENGTHS[strength]));

    // Weights (loaded from a file or trained here) are kept in this site's
    // IndexedDB, and in memory while in use. While the trainer runs it owns
    // them, so Learned play waits until training stops.
    let weights = null, haveWeights = false;
    let trainer = null, trainerUrl = '', stoppingTraining = false, lastSavedAt = '';

    // Learned modes need weights in memory before the first move.
    async function prepare() {
      if (STRENGTHS[strength].mode !== 'learned') return true;
      if (!weights) {
        render({ status: 'Loading weights…' });
        try {
          const blob = await idb('readonly', (store) => store.get('weights'));
          if (blob && isNetworkSize(blob.size / 4)) weights = await readWeights(blob);
        } catch (e) {
          weights = null;
        }
        if (!weights) {
          render({ status: 'No weights yet. Train or use "Load weights" first.' });
          return false;
        }
        render({ status: 'Ready', weights: SAVED });
      }
      // The engine may predate these weights (built for heuristic play, or before training).
      getEngine().setWeights(weights);
      return true;
    }

    let running = false, timer = 0, sawSavedGame = false, stuck = 0;
    let moves = 0, thinkMs = 0, speed = 'fast';

    // The game saves its state to localStorage synchronously on every move,
    // while the DOM only catches up on the next animation frame.
    function readSaved() {
      try {
        return JSON.parse(localStorage.getItem('gameState'));
      } catch (e) {
        return null;
      }
    }

    function stateFromSaved(saved) {
      const grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      saved.grid.cells.forEach((column, x) => column.forEach((cell, y) => {
        if (cell) grid[y][x] = cell.value;
      }));
      return { grid, score: saved.score, won: saved.won, keepPlaying: saved.keepPlaying };
    }

    // Fallback when storage is unavailable. Merged tiles leave their source
    // tiles in the DOM at the same position, so keep the largest value.
    function stateFromDom() {
      const grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      let found = false;
      tileContainer.querySelectorAll('.tile').forEach((el) => {
        const value = /(?:^|\s)tile-(\d+)(?:\s|$)/.exec(el.className);
        const pos = /tile-position-(\d)-(\d)/.exec(el.className);
        if (!value || !pos) return;
        const x = pos[1] - 1, y = pos[2] - 1;
        if (x < 0 || x > 3 || y < 0 || y > 3) return;
        grid[y][x] = Math.max(grid[y][x], Number(value[1]));
        found = true;
      });
      return found ? { grid, score: domScore(), won: false, keepPlaying: true } : null;
    }

    function domScore() {
      const el = document.querySelector('.score-container');
      return el ? parseInt(el.textContent, 10) || 0 : 0;
    }

    function pressKey(dir) {
      const code = KEY_CODES[dir];
      const ev = new KeyboardEvent('keydown', { key: KEY_NAMES[dir], code: KEY_NAMES[dir], bubbles: true, cancelable: true });
      // The game reads event.which, which synthetic events leave at 0.
      Object.defineProperty(ev, 'which', { get: () => code });
      Object.defineProperty(ev, 'keyCode', { get: () => code });
      document.dispatchEvent(ev);
    }

    // One move. Returns false when the loop should stop.
    function step() {
      if (messageEl.classList.contains('game-over')) return finish('Game over');

      const saved = readSaved();
      let state;
      if (saved && saved.grid && saved.grid.cells) {
        if (saved.grid.size !== 4) return finish('Only the 4x4 board is supported');
        sawSavedGame = true;
        state = stateFromSaved(saved);
      } else if (sawSavedGame) {
        return finish('Game over'); // the game deletes its save when it ends
      } else {
        state = stateFromDom();
        if (!state) return finish('Could not read the board');
      }

      // "You win!" at 2048 blocks moves until "Keep going" is pressed. The
      // message can also appear a frame after we already continued; clear it.
      if ((state.won && !state.keepPlaying) || messageEl.classList.contains('game-won')) {
        keepPlayingButton.click();
      }

      const result = getEngine().bestMove(state.grid);
      if (result.dir < 0) return finish('Game over');

      const before = saved ? JSON.stringify(saved) : null;
      pressKey(result.dir);
      if (before !== null && JSON.stringify(readSaved()) === before) {
        if (++stuck >= 3) return finish('Moves are not registering. Reload the page and try again.');
      } else {
        stuck = 0;
      }

      moves++;
      thinkMs += result.ms;
      const maxTile = Math.max(...state.grid.map((row) => Math.max(...row)));
      render({
        status: running ? 'Running' : 'Paused',
        stats: `Moves ${moves} · Score ${state.score} · Max tile ${maxTile}`,
        last: `${DIR_NAMES[result.dir]} · ${result.ms.toFixed(0)} ms · depth ${result.depthLimit} · avg ${(thinkMs / moves).toFixed(0)} ms`,
      });
      return true;
    }

    // Hidden tabs throttle setTimeout (down to once a minute in Chrome), so
    // yield through a MessageChannel there instead.
    const channel = new MessageChannel();
    channel.port1.onmessage = loop;

    function loop() {
      timer = 0;
      if (!running || !step() || !running) return;
      if (document.hidden) channel.port2.postMessage(null);
      else timer = setTimeout(loop, SPEEDS[speed]);
    }

    async function start() {
      if (running || !(await prepare())) return;
      running = true;
      stuck = 0;
      render({ status: 'Running' });
      loop();
    }

    function stop(status) {
      running = false;
      if (timer) clearTimeout(timer);
      timer = 0;
      render({ status: status || 'Paused' });
    }

    function finish(status) {
      stop(`${status} · final score ${domScore()}`);
      return false;
    }

    // ---- Panel -------------------------------------------------------------

    const panel = document.createElement('div');
    panel.id = 'algo2048-panel';
    panel.innerHTML = `
      <style>
        #algo2048-panel { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; width: 260px;
          padding: 12px; border-radius: 6px; background: #bbada0; color: #f9f6f2;
          font: 13px/1.4 "Clear Sans", "Helvetica Neue", Arial, sans-serif; box-shadow: 0 4px 16px rgba(0,0,0,.25); }
        #algo2048-panel strong { font-size: 15px; }
        #algo2048-panel .row { display: flex; gap: 6px; align-items: center; margin-top: 8px; }
        #algo2048-panel button, #algo2048-panel select { font: inherit; border: 0; border-radius: 3px; padding: 5px 10px;
          background: #8f7a66; color: #f9f6f2; cursor: pointer; }
        #algo2048-panel select { flex: 1; padding: 4px 6px; }
        #algo2048-panel .muted { color: #eee4da; font-size: 12px; }
      </style>
      <strong>2048 Algo</strong>
      <div class="row"><button data-act="toggle">Start</button><button data-act="step">Step</button></div>
      <div class="row">Speed <select data-set="speed">
        <option value="instant">Instant</option><option value="fast" selected>Fast</option><option value="watch">Watchable</option>
      </select></div>
      <div class="row">Search <select data-set="strength">
        <option value="best" selected>Best score</option><option value="quick">Quick</option>
        <option value="learned" disabled>Learned</option><option value="learned-deep" disabled>Learned (deep)</option>
      </select></div>
      <div class="row"><button data-act="weights">Load weights</button><button data-act="export">Export</button><button data-act="reset">Reset</button></div>
      <div class="muted" data-out="weights">No weights yet</div>
      <input type="file" accept=".bin" data-set="weights-file" hidden>
      <div class="row"><button data-act="train">Train</button><select data-set="rate">
        <option value="0.1" selected>Rate 0.1</option><option value="0.01">Rate 0.01</option><option value="0.001">Rate 0.001</option>
      </select></div>
      <div class="muted" data-out="training">Not trained in this browser yet</div>
      <div class="row" data-out="status">Ready</div>
      <div class="muted" data-out="stats"></div>
      <div class="muted" data-out="last"></div>`;
    document.body.appendChild(panel);

    const out = (name) => panel.querySelector(`[data-out="${name}"]`);
    const toggleButton = panel.querySelector('[data-act="toggle"]');
    const trainButton = panel.querySelector('[data-act="train"]');
    const rateSelect = panel.querySelector('[data-set="rate"]');

    function render(fields) {
      for (const [name, text] of Object.entries(fields)) out(name).textContent = text;
      toggleButton.textContent = running ? 'Stop' : 'Start';
      trainButton.textContent = !trainer ? 'Train' : stoppingTraining ? 'Saving…' : 'Stop training';
    }

    const strengthSelect = panel.querySelector('[data-set="strength"]');
    const fileInput = panel.querySelector('[data-set="weights-file"]');

    function setStrength(value) {
      strength = value;
      strengthSelect.value = value;
      if (engine) Object.assign(engine.options, STRENGTHS[strength]);
    }

    // Learned options need saved weights and no training in progress.
    function updateLearnedOptions() {
      const available = haveWeights && !trainer;
      strengthSelect.querySelectorAll('option[value^="learned"]').forEach((option) => {
        option.disabled = !available;
      });
      if (!available && STRENGTHS[strength].mode === 'learned') setStrength('best');
    }

    const count = (n) => Math.round(n).toLocaleString();
    const pct = (x) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
    const SAVED = 'Saved in this browser';

    // ---- Training in this tab ----------------------------------------------

    async function startTraining() {
      if (trainer) return;
      if (STRENGTHS[strength].mode === 'learned' && running) stop('Paused: Learned play waits while training');
      render({ training: 'Starting…' });

      let totals = { games: 0, steps: 0, seconds: 0 };
      try {
        if (!weights) {
          const blob = await idb('readonly', (store) => store.get('weights'));
          if (blob && isNetworkSize(blob.size / 4)) weights = await readWeights(blob);
        }
        totals = (await idb('readonly', (store) => store.get('stats'))) || totals;
      } catch (e) {
        // No browser storage: train from scratch; the first save reports the failure.
      }
      if (!weights) weights = newNetwork();
      if (Array.isArray(weights)) {
        render({ training: 'Large networks train in the dashboard on an NVIDIA GPU' });
        return;
      }

      try {
        trainerUrl = URL.createObjectURL(new Blob([trainingWorkerSource()], { type: 'text/javascript' }));
        trainer = new Worker(trainerUrl);
      } catch (e) {
        render({ training: `Can't train in this browser: ${e.message}` });
        return;
      }
      trainer.onmessage = onTrainerMessage;
      trainer.onerror = (e) => {
        // The worker held the only in-memory copy; the last save is still in storage.
        endTraining();
        render({ training: `Training crashed: ${e.message}. The last save is kept.` });
      };
      trainer.postMessage({ type: 'start', buffer: weights.buffer, alpha: Number(rateSelect.value), totals }, [weights.buffer]);
      weights = null;
      stoppingTraining = false;
      lastSavedAt = '';
      updateLearnedOptions();
      render({ training: `Training from ${count(totals.games)} games…` });
    }

    function stopTraining() {
      if (!trainer || stoppingTraining) return;
      stoppingTraining = true;
      trainer.postMessage({ type: 'stop' });
      render({ training: 'Saving…' });
    }

    function endTraining() {
      trainer.terminate();
      URL.revokeObjectURL(trainerUrl);
      trainer = null;
      stoppingTraining = false;
      updateLearnedOptions();
    }

    function onTrainerMessage(e) {
      const msg = e.data;
      if (msg.type === 'progress') {
        const r = msg.reach;
        render({
          training: `Training: ${count(msg.totals.games)} games, ${(msg.movesPerSecond / 1e6).toFixed(2)}M moves/s · ` +
            `new games avg ${count(msg.avg)}, 8192 ${pct(r[8192])}, 16384 ${pct(r[16384])}, 32768 ${pct(r[32768])}${lastSavedAt}`,
        });
      } else if (msg.type === 'saved') {
        haveWeights = true;
        lastSavedAt = ` · saved ${new Date().toLocaleTimeString()}`;
        render({ weights: SAVED });
      } else if (msg.type === 'error') {
        render({ training: `Save failed, still training: ${msg.error}` });
      } else if (msg.type === 'stopped') {
        weights = new Float32Array(msg.buffer);
        if (engine) engine.setWeights(weights);
        if (!msg.error) haveWeights = true;
        endTraining();
        render({
          training: msg.error
            ? `Stopped, but saving failed: ${msg.error}`
            : `Trained ${count(msg.totals.games)} games in this browser · saved`,
          weights: msg.error ? 'In memory only (saving failed)' : SAVED,
        });
      }
    }

    // Closing the tab mid-training loses progress since the last save (up to 5 minutes).
    const warnBeforeLeaving = (e) => {
      if (!trainer) return;
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', warnBeforeLeaving);

    async function exportWeights() {
      let blob = null;
      try {
        blob = await idb('readonly', (store) => store.get('weights'));
      } catch (e) {
        blob = null;
      }
      if (!blob) {
        render({ training: 'No saved weights to export yet' });
        return;
      }
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'network.bin';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 60000);
    }

    toggleButton.addEventListener('click', () => (running ? stop() : start()));
    panel.querySelector('[data-act="step"]').addEventListener('click', async () => {
      if (!running && (await prepare())) step();
    });
    panel.querySelector('[data-set="speed"]').addEventListener('change', (e) => {
      speed = e.target.value;
    });
    strengthSelect.addEventListener('change', (e) => setStrength(e.target.value));
    trainButton.addEventListener('click', () => (trainer ? stopTraining() : startTraining()));
    rateSelect.addEventListener('change', () => {
      if (trainer) trainer.postMessage({ type: 'alpha', alpha: Number(rateSelect.value) });
    });
    panel.querySelector('[data-act="export"]').addEventListener('click', exportWeights);
    panel.querySelector('[data-act="weights"]').addEventListener('click', () => {
      if (trainer) render({ weights: 'Stop training first' });
      else fileInput.click();
    });

    fileInput.addEventListener('change', async () => {
      const file = fileInput.files[0];
      fileInput.value = '';
      if (!file || trainer) return;
      if (file.size % 4 || !isNetworkSize(file.size / 4)) {
        render({ weights: `Not a weights file (needs ${[NETWORK_WEIGHTS, LARGE_NETWORK_WEIGHTS, XLARGE_NETWORK_WEIGHTS].map((n) => n * 4).join(', or ')} bytes)` });
        return;
      }
      render({ weights: 'Loading…' });
      weights = await readWeights(file);
      getEngine().setWeights(weights);
      let note = SAVED;
      try {
        // New weights: the browser's training totals no longer apply.
        await idb('readwrite', (store) => {
          store.delete('stats');
          return store.put(file, 'weights');
        });
      } catch (e) {
        note = 'For this visit only (browser storage failed)';
      }
      haveWeights = true;
      updateLearnedOptions();
      setStrength('learned');
      render({ weights: `${note} (${file.name})`, training: 'Not trained in this browser yet' });
    });

    panel.querySelector('[data-act="reset"]').addEventListener('click', async () => {
      if (trainer) {
        render({ weights: 'Stop training first' });
        return;
      }
      if (!confirm('Delete the weights saved in this browser? The next training starts a new network. Export first to keep a copy.')) return;
      try {
        await idb('readwrite', (store) => {
          store.delete('stats');
          return store.delete('weights');
        });
      } catch (e) {
        // Nothing stored, or no storage: clearing memory is enough.
      }
      weights = null;
      haveWeights = false;
      updateLearnedOptions();
      render({ weights: 'No weights yet', training: 'Not trained in this browser yet' });
    });

    idb('readonly', (store) => store.get('weights')).then((blob) => {
      if (!blob) return;
      if (!isNetworkSize(blob.size / 4)) {
        render({ weights: 'Saved weights are from an older version: Reset, or Load weights with a current file' });
        return;
      }
      haveWeights = true;
      updateLearnedOptions();
      render({ weights: SAVED });
    }, () => {});
    idb('readonly', (store) => store.get('stats')).then((stats) => {
      if (stats && stats.games && !trainer) render({ training: `Trained ${count(stats.games)} games in this browser` });
    }, () => {});

    window.algo2048 = {
      start,
      stop,
      async step() {
        if (!running && (await prepare())) step();
      },
      get engine() {
        return getEngine();
      },
      startTraining,
      stopTraining,
      destroy() {
        stop();
        stopTraining(); // the worker saves, then its 'stopped' message shuts it down
        channel.port1.close();
        window.removeEventListener('beforeunload', warnBeforeLeaving);
        panel.remove();
        delete window.algo2048;
      },
    };
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = {
      createEngine, createSelfPlay, trainingWorkerSource, networkLayout, isNetworkSize, clearLateEntries,
      DIR_NAMES, FEATURE_COUNT, NETWORK_TUPLES, NETWORK_LAYOUTS, NETWORK_WEIGHTS, LARGE_NETWORK_WEIGHTS,
      XLARGE_NETWORK_WEIGHTS, OPTIMISTIC_VALUE,
    };
  } else if (typeof document !== 'undefined') {
    installDriver();
  }
})();
