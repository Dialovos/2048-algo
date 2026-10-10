# 2048-algo

Plays [2048.org](https://www.2048.org/) for the highest score it can: an
expectimax search driven by either a hand-tuned heuristic or an n-tuple network
you train yourself on the CPU or an NVIDIA GPU. Version 1.8.

## What it can do

- **Plays in the browser.** A userscript adds a control panel to 2048.org and
  plays with real arrow-key events until no move is left, clicking **Keep
  going** at 2048.
- **Four search modes.** `Best score` is a hand-tuned heuristic at full depth
  (search and weights follow [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai)).
  `Quick` caps the depth for about 6× the speed. `Learned` uses a trained
  network 4 moves ahead; `Learned (deep)` looks 5 moves ahead, about 10× slower.
- **Exact boards up to 131072.** Four 20-bit rows, with moves and scores
  precomputed for all 2^20 rows.
- **Trains its own network.** TD(0) self-play with optimistic starting values,
  per-weight adaptive learning rates, and restarts from saved late-game boards.
- **Three network sizes.** Standard (4 patterns, 268 MB), large (8 patterns × 3
  game stages plus a corner bonus, 1.6 GB) and extra large (5 stages aimed at
  65536, 2.7 GB). Large ones train on the GPU only; every size plays in the
  browser on the CPU.
- **Three places to train.** Every CPU core (Node worker threads), an NVIDIA GPU
  (CUDA through CuPy, 5–7× the CPU's speed), or one core inside the game tab.
  CPU and GPU read and write the same files, so a network can switch devices
  between runs.
- **Local dashboard.** Start and stop training, watch progress and history
  charts, and download weights at http://127.0.0.1:2048.

| Setting | Result (small samples) |
| --- | --- |
| `Best score` heuristic | 16384, 387,388 points (1 game) |
| Standard network, GPU-trained, 4-move search | 16384 in 10 of 12 games, none reached 32768 |
| Large network after 1 hour of GPU training, 4-move search | 32768 in 16 of 24 games |

No trained weights ship with the repo; the Learned modes unlock once you train
or load a network.

## Run it

### Play

1. Install `2048-algo.user.js` in Tampermonkey or Violentmonkey, or paste the
   file into the DevTools console on 2048.org.
2. Press **Start** in the panel at the bottom right.

The panel also has **Step**, **Speed**, **Search**, **Load weights** (pick a
`.bin`; the browser keeps it in IndexedDB, so once per network), **Export**,
**Reset**, and **Train** with its **Rate**. From the console:
`algo2048.start()`, `algo2048.stop()`, `algo2048.step()`, `algo2048.destroy()`.
Keep the tab open; a tab the browser freezes or discards stops playing.

### Train

Needs Node.js (22 tested).

```sh
node dashboard.js                                   # dashboard at http://127.0.0.1:2048; Ctrl+C saves and quits
node train.js                                       # train or resume weights/network.bin on every core
node train.js --weights weights/mine.bin --threads 6 --hours 8
node train.js --backend gpu --gpu-games 1024 --weights weights/mine.bin
node train.js --backend gpu --layout large --from weights/mine.bin --weights weights/mine-large.bin
```

Ctrl+C saves; running again resumes. Every option is listed at the top of
`train.js`. A network lives in `weights/` as `NAME.bin` (weights), `NAME.tc`
(adaptive-rate state), `NAME.json` (totals) and logs, all gitignored. To play
it, press **Download weights** in the dashboard (or take the `.bin` directly)
and pick the file with the panel's **Load weights**.

### NVIDIA GPU (optional)

Needs an NVIDIA driver with CUDA 13 and Python 3.12:

```sh
uv venv --python 3.12
uv pip install -r requirements-gpu.txt
.venv/bin/python train-gpu.py --probe               # Windows: .venv/Scripts/python.exe
```

CuPy's wheels bring the CUDA libraries, so no separate CUDA Toolkit is needed.
The dashboard finds `.venv` on its own; `ALGO2048_PYTHON` points it at another
Python. An extra-large network needs about 9 GB of GPU memory to start. CPU training
needs neither Python nor CuPy.

### Evaluate and test

```sh
node simulate.js --games 3 --seed 1                                     # heuristic games
node simulate.js --weights weights/mine.bin --search-depth 4 --games 8  # a trained network
node simulate.js --verify-only                                          # engine vs a plain reference
node --test tests/test-dashboard.js                                     # dashboard; GPU half skips without one
.venv/bin/python -m unittest discover -s tests -p test_gpu.py           # CUDA vs JS parity
node benchmark-training.js --seconds 20 --gpu-games 1024                # CPU vs GPU training speed
```
