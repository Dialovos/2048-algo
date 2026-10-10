#!/usr/bin/env python3
"""CUDA self-play. JSON-lines control/progress for gpu.js; checkpoints match train.js."""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".venv" / ".cupy-cache"))
BLOCK = 1 << 24  # weights per pattern
# Same patterns and layouts as the JS engine: standard networks use the first
# four patterns in one stage; large and extra-large ones use all eight in each
# of 3 or 5 stages (plus a corner bonus). Layout name -> (patterns, stages).
TUPLES = ((0, 1, 2, 3, 4, 5), (4, 5, 6, 7, 8, 9), (0, 1, 2, 4, 5, 6), (4, 5, 6, 8, 9, 10),
          (0, 1, 2, 3, 4, 8), (0, 1, 2, 4, 5, 8), (0, 1, 4, 5, 6, 7), (0, 1, 2, 5, 6, 7))
LAYOUTS = {"standard": (4, 1), "large": (8, 3), "xlarge": (8, 5)}
NW = 4 * BLOCK  # standard
CHUNK = 1 << 24  # floats per host transfer, keeping host memory bounded
# Restart pool codes from the kernel -> (label, rank a restarted game must reach).
# Labels start with the tile the dashboard doubles for its target.
RESTARTS = {
    "standard": {13 + k: (str(2 ** (13 + k)), 14 + k) for k in range(5)},
    "large": {13: ("8192", 14), 14: ("16384", 15), 15: ("16384+8192", 15), 16: ("32768", 16), 17: ("65536", 17)},
    "xlarge": {13: ("8192", 14), 14: ("16384", 15), 15: ("16384+8192", 15), 16: ("32768", 16),
               17: ("32768+8192", 16), 18: ("65536", 17)},
}


def emit(kind, **data):
    print(json.dumps({"type": kind, **data}, allow_nan=False), flush=True)


def feature_cells(patterns):
    result = []
    for pattern in TUPLES[:patterns]:
        for rotation in range(4):
            for flip in range(2):
                for cell in pattern:
                    r, c = divmod(cell, 4)
                    if flip:
                        c = 3 - c
                    for _ in range(rotation):
                        r, c = c, 3 - r
                    result.append(r * 4 + c)
    return result


def layout_of(path):
    sizes = {patterns * stages * BLOCK * 4: name for name, (patterns, stages) in LAYOUTS.items()}
    size = path.stat().st_size
    if size not in sizes:
        raise ValueError(f"{path.name} is not a weights file ({', '.join(map(str, sizes))} bytes).")
    return sizes[size]


def source_stage(stage, source_stages):
    """The stage of a starting network whose weights a new network's stage copies:
    the one the source used for the same boards (a 3-stage network plays 32768
    boards with its 16384 stages)."""
    if stage == 0 or source_stages == 1:
        return 0
    level = min((stage - 1) // 2, (source_stages - 1) // 2 - 1)
    return 1 + 2 * level + (stage - 1) % 2


class GpuTrainer:
    """layout and source apply only when creating a network. source is a smaller
    network to start from: each stage copies the source's weights for the same
    boards (patterns it lacks start at 0), so the new network plays like it at
    first. Each stage also copies that source stage's adaptive-rate state,
    except late stages copied from a standard network's single stage: those
    weights never specialised in the late game, so they start at full rates.
    (Fresh rates on already-specialised weights let noisy late-game episodes
    overwrite them: an extra-large network made that way from gamma-large lost
    half its 16384 -> 32768 rate in 5 minutes.)"""

    def __init__(self, weights=None, *, games=1024, tc=True, optimism=320000, seed=2048,
                 alpha=None, restart=0.5, capacity=8192, exact=False, layout="standard", source=None):
        import cupy as cp
        import numpy as np
        self.cp, self.np = cp, np
        self.n, self.capacity = games, capacity
        self.restart = restart
        path = Path(weights) if weights else None
        exists = bool(path and path.exists())
        tc_path = path.with_suffix(".tc") if path else None
        source = Path(source) if source else None
        if exists and not tc and tc_path.exists():
            raise ValueError("This network has adaptive-rate state; omit --no-tc to preserve it.")
        if not exists and tc_path and tc_path.exists():
            raise ValueError("An adaptive-state file already exists without its weights; choose another network name.")
        if source and exists:
            raise ValueError("A starting network only applies when creating a new one.")
        self.layout = layout_of(path) if exists else layout
        self.patterns, self.stages = LAYOUTS[self.layout]
        self.levels = (self.stages - 1) // 2
        self.nw = self.patterns * self.stages * BLOCK
        if source:
            source_patterns, source_stages = LAYOUTS[layout_of(source)]
            if source_stages >= self.stages:
                raise ValueError("Start a network from one with fewer stages.")
        self.use_tc = bool(tc and (not exists or tc_path.exists()))
        self.alpha = float(alpha if alpha is not None else (1 if self.use_tc else 0.1))
        # exact=True rounds weight updates exactly as the JS trainer does (parity
        # tests); training uses hardware float atomics, which are several times faster.
        options = ("--std=c++17", "--fmad=false", f"-DLEVELS={self.levels}") \
            + (("-DEXACT_ATOMICS",) if exact else ())
        self.module = cp.RawModule(code=(ROOT / "gpu-kernels.cu").read_text(), options=options)
        self.module.compile()
        feature_list = feature_cells(self.patterns)
        cells = cp.ndarray((len(feature_list),), dtype=cp.int32, memptr=self.module.get_global("FEATURES"))
        cells.set(np.array(feature_list, dtype=np.int32))
        self.left, self.right, self.rewards = [cp.empty(1 << 20, dtype=cp.uint32) for _ in range(3)]
        self.module.get_function("row_tables")((4096,), (256,), (self.left, self.right, self.rewards))

        def read(file, target, offset=0):
            # Fill a device array from little-endian float32s at a float offset, in bounded chunks.
            for start in range(0, target.size, CHUNK):
                count = min(CHUNK, target.size - start)
                target[start:start + count] = cp.asarray(
                    np.fromfile(file, dtype="<f4", count=count, offset=(offset + start) * 4))

        if exists:
            self.w = cp.empty(self.nw, dtype=cp.float32)
            read(path, self.w)
        elif source:
            self.w = cp.zeros(self.nw, dtype=cp.float32)
            for stage in range(self.stages):
                first = stage * self.patterns * BLOCK
                read(source, self.w[first:first + source_patterns * BLOCK],
                     source_stage(stage, source_stages) * source_patterns * BLOCK)
        else:
            self.w = cp.empty(self.nw, dtype=cp.float32)
            self.module.get_function("initialize_weights")((self.nw // 256,), (256,), (self.w, np.float32(optimism)))
        self.e, self.a = [cp.zeros(self.nw if self.use_tc else 1, dtype=cp.float32) for _ in range(2)]
        if self.use_tc and exists:
            if tc_path.stat().st_size != self.nw * 8:
                raise ValueError("Adaptive-rate state must hold two float32 arrays the size of the weights.")
            read(tc_path, self.e)
            read(tc_path, self.a, self.nw)
        elif self.use_tc and source and source.with_suffix(".tc").exists():
            source_nw = source_patterns * source_stages * BLOCK
            for stage in range(self.stages):
                from_stage = source_stage(stage, source_stages)
                if stage and not from_stage:
                    continue  # a late stage copied from a standard network: full rates
                first, offset = stage * self.patterns * BLOCK, from_stage * source_patterns * BLOCK
                part = slice(first, first + source_patterns * BLOCK)
                read(source.with_suffix(".tc"), self.e[part], offset)
                read(source.with_suffix(".tc"), self.a[part], source_nw + offset)
        if not self.finite():
            raise ValueError("Checkpoint contains invalid weights or adaptive-rate state.")
        self.state = cp.zeros((12, games), dtype=cp.uint32)
        seeds = (np.arange(games, dtype=np.uint32) + np.uint32(seed))
        seeds[seeds == 0] = 1
        self.state[10].set(seeds)
        self.targets = cp.zeros(games, dtype=cp.float64)
        self.history = cp.empty((capacity, 5, games), dtype=cp.uint32)
        pools = 6 if self.levels == 2 else 5
        self.pools = cp.empty((pools, 1000, 4, games), dtype=cp.uint32)
        self.counts = cp.zeros((pools, games), dtype=cp.uint32)
        self.results = cp.empty((5, games), dtype=cp.uint32)
        self.moves = cp.zeros(games, dtype=cp.uint64)
        self.overflow = cp.zeros(1, dtype=cp.int32)
        self.kernel = self.module.get_function("advance")
        cp.cuda.get_current_stream().synchronize()

    def advance(self, transitions=64):
        cp, np = self.cp, self.np
        self.kernel(((self.n * 32 + 127) // 128,), (128,), (
            self.w, self.e, self.a, np.int32(self.use_tc), self.left, self.right, self.rewards,
            self.state, self.targets, self.history, self.pools, self.counts, self.results,
            self.moves, self.overflow, np.int32(self.n), np.int32(self.capacity), np.int32(transitions),
            np.float64(self.alpha), np.float64(self.restart)))
        results = self.results.get()
        needed = int(self.overflow.get()[0])
        if needed:
            # Never truncate a long game or train against an artificial terminal.
            capacity = max(needed, self.capacity * 2)
            expanded = cp.empty((capacity, 5, self.n), dtype=cp.uint32)
            expanded[:self.capacity] = self.history
            self.history = expanded
            self.capacity = capacity
            del expanded
            cp.get_default_memory_pool().free_all_blocks()  # return the old history to the GPU
            self.overflow.fill(0)
        return results[1:, results[0] != 0].T

    def finite(self):
        # Chunked, so checking a 2.7 GB network needs no weights-sized temporaries.
        cp = self.cp
        for start in range(0, self.nw, CHUNK):
            part = slice(start, start + CHUNK)
            if not bool(cp.isfinite(self.w[part]).all()) or (self.use_tc and not (
                    bool(cp.isfinite(self.e[part]).all()) and bool((cp.isfinite(self.a[part]) & (self.a[part] >= 0)).all()))):
                return False
        return True

    def save(self, path, totals):
        path = Path(path)
        if not self.finite():
            raise ValueError("Non-finite training state; the last saved checkpoint was preserved.")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = [(path, [self.w])]
        if self.use_tc:
            arrays.append((path.with_suffix(".tc"), [self.e, self.a]))
        for destination, sources in arrays:
            tmp = destination.with_name(destination.name + ".tmp")
            with tmp.open("wb") as f:
                for source in sources:
                    # Keep host memory bounded when checkpointing.
                    for start in range(0, source.size, 1 << 22):
                        source[start:start + (1 << 22)].get().astype("<f4", copy=False).tofile(f)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(destination)
        destination = path.with_suffix(".json")
        tmp = destination.with_name(destination.name + ".tmp")
        tmp.write_text(json.dumps(totals, indent=2) + "\n")
        tmp.replace(destination)


def probe():
    import cupy as cp
    count = cp.cuda.runtime.getDeviceCount()
    if not count:
        raise RuntimeError("No NVIDIA CUDA GPU found.")
    # Exercise compilation as well as driver enumeration.
    check = cp.RawKernel('extern "C" __global__ void check(int* x) { x[0]=42; }', "check")
    x = cp.zeros(1, dtype=cp.int32)
    check((1,), (1,), (x,))
    if int(x.get()[0]) != 42:
        raise RuntimeError("CUDA kernel check failed.")
    info = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    return {"available": True, "name": info["name"].decode(), "freeBytes": free, "totalBytes": total}


def benchmark(args):
    gpu = GpuTrainer(args.weights, games=args.games, tc=not args.no_tc, optimism=args.optimism,
                     seed=args.seed, alpha=args.alpha, restart=args.restart, layout=args.layout, source=args.source)
    warm = time.monotonic()
    while time.monotonic() - warm < 2:
        gpu.advance()
    start, games, moves = time.monotonic(), 0, 0
    while time.monotonic() - start < args.benchmark:
        completed = gpu.advance()
        games += len(completed)
        moves += int(completed[:, 3].sum())
    elapsed = time.monotonic() - start
    emit("benchmark", backend="gpu", parallel=args.games, tc=gpu.use_tc, layout=gpu.layout,
         seconds=elapsed, games=games, moves=moves, movesPerSecond=moves/elapsed)


def run(args):
    commands = queue.Queue()
    def read_commands():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except ValueError:
                pass
        # Losing the dashboard must save and stop, not orphan GPU work.
        if args.control:
            commands.put({"command": "stop"})
    if args.control:
        threading.Thread(target=read_commands, daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: commands.put({"command": "stop"}))
    path = Path(args.weights)
    existing = path.exists()
    totals = {"games": 0, "steps": 0, "seconds": 0}
    if existing and path.with_suffix(".json").exists():
        totals.update(json.loads(path.with_suffix(".json").read_text()))
    gpu = GpuTrainer(path, games=args.games, tc=not args.no_tc, optimism=args.optimism,
                     seed=args.seed, alpha=args.alpha, restart=args.restart, layout=args.layout, source=args.source)
    started = time.monotonic()
    previous_seconds = totals["seconds"]
    last_report, last_save = started, started
    fresh = deque(maxlen=10000)
    restarts = {}
    sample_games, sample_moves, sample_time = totals["games"], 0, started
    log_path, history_path = path.with_suffix(".log"), path.with_suffix(".history.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        with log_path.open("a") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message + "\n")
        emit("log", message=message)

    def summary():
        nonlocal sample_games, sample_moves, sample_time
        now = time.monotonic()
        moves = int(gpu.moves.sum().get())
        seconds = max(now - sample_time, 1e-6)
        result = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed": now - started, "games": totals["games"], "steps": totals["steps"],
            "gamesPerSecond": (totals["games"] - sample_games) / seconds,
            "movesPerSecond": (moves - sample_moves) / seconds,
            "newGames": len(fresh), "avg": sum(s for s, _ in fresh) / max(1, len(fresh)),
            "reach": {str(t): sum(tile >= t for _, tile in fresh) / max(1, len(fresh))
                      for t in (2048, 4096, 8192, 16384, 32768)},
            "restarts": {label: sum(values) / len(values) for label, values in restarts.items()},
        }
        sample_games, sample_moves, sample_time = totals["games"], moves, now
        return result

    def save():
        totals["seconds"] = previous_seconds + time.monotonic() - started
        gpu.save(path, totals)
        log("Checkpoint saved.")

    emit("ready", tc=gpu.use_tc, alpha=gpu.alpha, games=args.games, layout=gpu.layout, totals=totals)
    log(f"GPU {'resumed' if existing else 'started'} {path.name}; {gpu.layout} network "
        f"({gpu.patterns} patterns x {gpu.stages} stages{', corner bonus' if gpu.levels else ''}); {args.games} parallel games; "
        f"{'TC learning' if gpu.use_tc else 'fixed rate'}; step {gpu.alpha}.")
    if args.source:
        log(f"Every stage starts from {Path(args.source).name}'s weights and adaptive rates for the same boards"
            f"{'' if Path(args.source).with_suffix('.tc').exists() else ' (it has no adaptive state)'}; "
            f"late stages copied from a standard network start at full rates.")
    if existing and not gpu.use_tc:
        log("No adaptive-rate state loaded; continuing these weights with a fixed learning rate.")
    try:
        while True:
            for start, score, tile, steps in gpu.advance():
                totals["games"] += 1
                totals["steps"] += int(steps)
                if start == 0:
                    fresh.append((int(score), int(tile)))
                else:
                    label, need = RESTARTS[gpu.layout][int(start)]
                    restarts.setdefault(label, deque(maxlen=1000)).append(int(tile) >= 2 ** need)
            now = time.monotonic()
            if now - last_report >= args.report_seconds:
                s = summary()
                emit("summary", summary=s)
                log(f"{s['games']:,} games; {s['movesPerSecond']/1e6:.2f}M moves/s; "
                    f"new-game average {s['avg']:.0f}.")
                with history_path.open("a") as f:
                    f.write(json.dumps(s) + "\n")
                last_report = now
            if now - last_save >= args.save_minutes * 60:
                save()
                last_save = time.monotonic()
            if now - started >= args.seconds:
                commands.put({"command": "stop"})
            while not commands.empty():
                command = commands.get()
                if command.get("command") in ("save", "stop"):
                    save()
                    emit("saved", id=command.get("id"))
                    last_save = time.monotonic()
                    if command["command"] == "stop":
                        emit("summary", summary=summary())
                        return
    except Exception:
        # A failed checkpoint must never be reported as a successful stop.
        log("Training failed; attempting to preserve the current finite weights.")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--benchmark", type=float, default=0, metavar="SECONDS", help="In-memory timing; never save or change a checkpoint")
    parser.add_argument("--weights", default="weights/network.bin")
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--restart", type=float, default=0.5)
    parser.add_argument("--optimism", type=float, default=320000)
    parser.add_argument("--no-tc", action="store_true")
    parser.add_argument("--layout", choices=LAYOUTS, default="standard",
                        help="New network: standard (4 patterns); large (8 patterns x 3 stages, corner bonus); xlarge (5 stages)")
    parser.add_argument("--from", dest="source", metavar="FILE", help="New network: start from this smaller network")
    parser.add_argument("--seconds", type=float, default=math.inf)
    parser.add_argument("--report-seconds", type=float, default=5)
    parser.add_argument("--save-minutes", type=float, default=10)
    args = parser.parse_args()
    if args.probe:
        emit("capability", **probe())
        return
    if not 32 <= args.games <= 8192 or not 0 <= args.restart <= 0.9:
        parser.error("Use 32–8192 parallel games and a restart share between 0 and 0.9.")
    if args.alpha is not None and not 0 < args.alpha <= 1:
        parser.error("Step size must be above 0 and at most 1.")
    if not (args.seconds > 0 and args.report_seconds > 0 and args.save_minutes > 0):
        parser.error("Run, report, and save intervals must be positive.")
    if args.benchmark:
        if not math.isfinite(args.benchmark) or args.benchmark <= 0:
            parser.error("Benchmark duration must be positive and finite.")
        benchmark(args)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("error", message=str(error))
        sys.exit(1)
