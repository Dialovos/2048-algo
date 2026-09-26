# 2048-algo

- **Purpose:** An algorithm that plays [2048.org](https://www.2048.org/) for the highest score it can, using an expectimax search with either a hand-tuned heuristic or a trained n-tuple network.
- **Run:** Install `2048-algo.user.js` in Tampermonkey or Violentmonkey, or paste the file into the DevTools console on the game page, then press **Start** in the panel at the bottom right.
- **Train:** `node dashboard.js` opens a local training dashboard at http://127.0.0.1:2048 that trains on this computer's cores and hands out weights for the panel (see [Training](#training)). `node train.js` does the same from the command line.
- **Test:** `node simulate.js` (Node 18+, no dependencies). It checks moves, points, and big-tile boards, then plays full games headless. Add `--weights weights/NAME.bin --search-depth 4` to measure a trained network.
- **Status:** Working (2026-09-24, version 1.6). No mode has reached 32768 in testing yet. The best trained network (`weights/alpha-single.bin`) reaches 16384 in about half its games at 4-move search but fails in the late game; heuristic mode (`Best score`) is slower but, as a port of nneonneo/2048-ai, should be the stronger late-game player (see [Where things stand](#where-things-stand)). The panel and dashboard were tested in jsdom and over HTTP; not yet confirmed in a real browser.

## Using it

The panel has:

- **Start / Stop**: play continuously. **Step** plays one move.
- **Speed**: pause between moves. `Instant` runs as fast as the search allows; `Watchable` is easy to follow.
- **Search**:
  - `Best score` (default): hand-tuned heuristic at full depth.
  - `Quick`: the same heuristic with the depth capped, about 6x faster and somewhat weaker.
  - `Learned`: the trained network, looking 4 moves ahead.
  - `Learned (deep)`: the network looking 5 moves ahead; about 10x slower than `Learned`.
- **Load weights**: pick a network's `.bin` file (268 MB, from the dashboard, `train.js`, or **Export**). It is saved in the browser (IndexedDB for 2048.org), so this is needed once per network, or again after clearing site data. The Learned options unlock once weights are saved.
- **Export**: downloads the browser's saved weights as `network.bin`.
- **Reset**: after asking, deletes the browser's saved weights and training totals, so the next **Train** starts a new network.
- **Train / Stop training** and **Rate**: train in this tab on one CPU core (see [Training](#training)).

It clicks **Keep going** at 2048 and plays until no move is left. From the console you can also call `algo2048.start()`, `algo2048.stop()`, `algo2048.step()` and `algo2048.destroy()`. The engine builds its tables (about 50 MB) on the first move; Learned mode also loads the weights into memory.

Keep the tab open. In a background tab it switches to a message-channel loop so Chrome's timer throttling doesn't stall it, but a tab the browser freezes or discards stops.

## Training

### Dashboard

```sh
node dashboard.js             # opens http://127.0.0.1:2048; Ctrl+C saves any training and quits
```

- **Network:** pick one from `weights/`, or **New network…** with a name (lowercase letters, digits, dashes).
- **Run:** choose how many CPU cores to use (1 trains one game at a time; more run games in parallel), then **Start training**. **Save now** writes a checkpoint; **Stop and save** ends the run. It also autosaves every 10 minutes.
- **New-network options:** **Optimistic start** and **Adaptive learning rates** apply when a network is created and are on by default (see [How training works](#how-training-works)). **Step size** can stay blank for automatic, and the **restart share** sets how many games start from saved late-game boards.
- **Progress:** games trained, speed, the new-game average score, how often new games reach each tile, and how often restarted games make the next tile. **History** charts the average score and the 16384 and 32768 rates across all sessions of the selected network.
- **Use on 2048.org:** **Download weights** saves the selected network's file (while training, it's the last save, so press **Save now** first). Alternatively, **Copy file path** and pick that file in the panel's **Load weights**. The card also links to the userscript for installing or updating it.
- The server only listens on 127.0.0.1 and rejects requests from other websites. Restart it after updating the code.

### Command line

```sh
node train.js                              # new or resumed weights/network.bin; Ctrl+C saves and quits
node train.js --weights weights/other.bin  # another network
node train.js --threads 6 --hours 8        # leave two cores free; stop on its own
```

See the top of `train.js` for every option, including `--no-tc` and `--no-optimistic`.

### In the browser

The panel's **Train** trains in a background worker in the 2048.org tab: no terminal, but one CPU core (the site doesn't allow the shared memory that parallel training needs), so it runs at about a quarter of the dashboard's speed. It starts from the browser's saved weights (or a new network), saves every 5 minutes and when stopped, and the browser asks before closing the tab mid-training. Learned play waits while it trains, because the worker holds the weights. **Rate** is its fixed learning rate: start at 0.1 and lower it only when progress stalls, since lower rates learn much more slowly.

### Files

The trained `weights/alpha-single.bin` (256 MiB) is versioned with Git LFS. With Git LFS installed, run `git lfs pull` after cloning or pulling this repository to download it, then select the file with **Load weights** in the panel.

Each network is a set of files in `weights/`: `NAME.bin` (the weights, 268 MB), `NAME.tc` (adaptive-rate state, 536 MB), `NAME.json` (totals), `NAME.log` (every progress line and conversion note), and `NAME.history.jsonl` (the dashboard's charts). Only `alpha-single.bin` is tracked; other networks, training state, and logs remain gitignored.

When using the workspace's shared Git hooks, upload committed weight updates with `git lfs push origin HEAD` before pushing the branch.

### Reading the progress

- **New games** start from an empty board and look one move ahead, so they measure the network without search. Search can change the picture a lot either way: an older plain network reached 16384 in 1% of greedy games but 50% at 4-move search, while `alpha-single` reaches it in about 33% greedy and 50% with search.
- To measure a network with search: `node simulate.js --weights weights/NAME.bin --search-depth 4 --games 14`.
- **Restarts** (for example `16384 -> 32768`) show how often games started from a saved late-game board make the next tile. That is the late-game skill that 32768 needs, and it has stayed at 0% so far.

## How training works

- **TD(0) self-play:** each game is played greedily on the current weights, then every afterstate's value is moved toward the points of the next move plus the next afterstate's value, working back from the end. Training threads update one shared weight table without locks.
- **Restarts:** half the games start from boards where an earlier game first made 8192, 16384 or 32768, so late-game positions get practice that full games rarely reach.
- **Optimistic start (new networks, default on):** every weight starts at 320,000 / 32, so positions training hasn't tried look attractive and get explored. Patterns holding 16384 or more start at 0 instead: they only occur in rare late games, and optimistic values there pulled the big tile toward the middle of the board. When choosing moves (not in training), a weight still at its starting value counts as 0, because search reaches rare positions whose untouched optimistic weights would otherwise flatter them.
- **Adaptive rates (TC learning, default on, dashboard and `train.js` only):** each weight keeps a running sum of its errors (E) and of their sizes (A) and scales its step by |E| / A, so rates shrink on their own and no 0.1 → 0.01 schedule is needed.
- **In-browser trainer:** a Web Worker built from this script's own engine and self-play functions (their source text plus the constants they use). The page transfers the weights to the worker, which trains in 100 ms chunks, saves a copy to IndexedDB every 5 minutes, and transfers the weights back when stopped.

### What was tried on 2026-09-24

- **Optimistic start plus adaptive rates** clearly helped the early and middle game. After 4.2M games (about 2 hours on 7 cores), the `alpha` network's new games averaged 169k and reached 8192 in 82%, against about 112k and 57% for the old method after 20 minutes.
- **Separate late-game weights (versions 1.4–1.5) made things worse and were removed.** Replacing the weights once a board held 16384 left the late game starting from nothing, so the 16384 drifted out of the corner and games died within about 2,000 moves. Adding an adjustment on top instead still drifted, and after 30 minutes the network reached 16384 in 0 of 14 searched games. As a single set of weights, `alpha`'s trained stage went back to 33% for 16384 in greedy games, and after 16384 it survived a median of about 3,700 moves.
- The original `alpha` files are in `Archive/2026-09-24-2048-algo-alpha-late-reset/`. `weights/alpha.bin` is a two-stage file this version can't use; `weights/alpha-single.bin` replaces it.

## How it works

- **Reading the board:** the game saves its full state to `localStorage["gameState"]` on every move. It saves before the tiles finish animating, so the algorithm can move as soon as it has decided. If storage is unavailable it reads the tile elements instead.
- **Playing:** it dispatches `keydown` arrow-key events on `document`, the same input the game handles.
- **Board encoding:** four 20-bit rows with 5 bits per tile rank, so every tile up to 131072 (the largest a 4x4 board can hold) is exact. Moves, points, and heuristic scores come from tables precomputed for all 2^20 rows, and columns are handled by transposing.
- **Search:** expectimax. It takes the best of its four moves; the new tile is averaged over every empty cell (2 at 90%, 4 at 10%). Branches less likely than 0.01% are scored without further search, and a transposition table caches positions within one decision.
- **Heuristic mode:** depth grows with the number of distinct tiles (`max(3, distinct - 2)`). The heuristic rewards empty cells, mergeable neighbours, and rows and columns whose tile ranks rise or fall steadily, and penalizes large tiles in general, which pushes it to merge them. The search, heuristic, and weights follow [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai).
- **Loss margin (heuristic mode):** a lost board scores 0. With tiles of 65536 and up, the large-tile penalty can push a live board's score toward or below 0, which would make losing look acceptable. When the current board scores below 300,000, the search lifts every leaf score by the difference. In simulated games the board scored above about 800,000 in 99% of positions up to 16384, so the lift only applies to rare messy boards and to big-tile endgames.
- **Learned mode:** the network estimates the points still to come after a move (before the new tile appears). It reads four 6-cell patterns, cells `0-5`, `4-9`, `0-2 + 4-6` and `4-6 + 8-10` numbered row by row, in all 8 rotations and reflections of the board, and sums 32 weights. The search scores a move as its points plus the value that follows, and a lost board has nothing left to score, so no loss penalty is needed. Ranks are capped at 15 inside the network, so it sees tiles from 32768 up as alike; the board itself stays exact.
- **Weights file:** 4 × 16^6 little-endian float32 values (268,435,456 bytes), pattern by pattern. A pattern's index packs its six cells' ranks at 4 bits each, first cell lowest. Changing the patterns or their order breaks existing files. Two-stage files (536,870,912 bytes) from versions 1.4–1.5 are not supported.

## Ceiling

- **Game rules:** the largest possible tile is 131072 and the largest possible score is 3,932,156. That needs a full board of distinct tiles from 4 to 131072, and 4s spawning at exactly the right moments.
- **Best published program:** Guei et al. (2021) use this kind of network with more techniques, much more training, and deeper search. As I recall, they report 32768 in about 72% of games and 65536 in about 0.02%, so 65536 stays extremely rare even at the state of the art. As far as I know, no program has made 131072.
- **Final score by largest tile:** roughly 210k–400k when the game ends with 16384, 460k–870k with 32768, and 1.0M–1.9M with 65536.

## Where things stand

- **For 32768 now, try `Best score`.** It is the nneonneo/2048-ai search, which that project reports reaching 32768 in about a third of games (as I recall). Here it has only been run once, reaching 16384 with 387,388 points. It is slow: about 0.16 s per move on average and seconds late in a game, so a 32768 game would take a couple of hours in the browser.
- **Learned play is fast but stalls in the late game.** Its restarted games from 16384 have not made 32768 yet. More training on `alpha-single` is the next step; it was still improving when stopped (new-game average 158k → 196k in its last 30 minutes).

## Simulated results

`node simulate.js --games N --seed S` plays with the real spawn rules. Add `--depth-bias -1 --max-depth 5` to match the Quick setting, or `--weights FILE --search-depth N` for Learned.

| Setting | Games | Reached 16384 | Scores | Time per move |
| --- | --- | --- | --- | --- |
| Best score (heuristic) | 1 (seed 7) | 1 of 1 | 387,388 | 159 ms average, 4.4 s slowest |
| Quick (heuristic) | 3 (seed 3) | 2 of 3 (the other 8192) | 158,444 to 343,136 | 20 to 29 ms average |
| Older browser network, 3-move search | 16 (seeds 201–202) | 7 of 16 | 60,484 to 339,196 (average about 216k) | under 1 ms average |
| Older browser network, 4-move search | 12 (seeds 101–104) | 6 of 12 | 168,316 to 361,848 | 7 to 10 ms average |
| `alpha-single`, 4-move search | 14 (seeds 601–607) | 7 of 14 | 157,304 to 346,108 (average 246,479) | 11 ms average |

None of these games reached 32768. The heuristic games ran on version 1.0; the older browser network is `weights/browser-2026-09-24.bin` (plain TD, fixed rate). These are small samples, so treat them as a rough guide.
