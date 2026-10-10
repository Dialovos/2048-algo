"""Run with .venv/bin/python -m unittest discover -s tests -p test_gpu.py -v."""
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_gpu", ROOT / "train-gpu.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(request):
    return json.loads(subprocess.check_output(["node", "tests/cpu-reference.js"],
        input=json.dumps(request), text=True, cwd=ROOT))


def rows(grid):
    return [sum((int(tile).bit_length() - 1 if tile else 0) << (5*c)
                for c, tile in enumerate(row)) for row in grid]


def initial_weights(indexes, layout):
    """Starting weights as the GPU and the reference set them (optimism 320000)."""
    extra = ((indexes >> 24) % 8 >= 4) if layout != "standard" else cp.zeros(indexes.shape, dtype=bool)
    return cp.where(extra | (indexes & (indexes >> 1) & (indexes >> 2) & 0x222222 != 0), 0, 10000)


# Late-game starts: 16384 in a corner (stage 2), and with an 8192 beside it
# (stage 3); likewise 32768 (stages 4 and 5 of an extra-large network).
STAGE_2 = [[16384, 1024, 256, 4], [512, 128, 16, 2], [64, 8, 2, 0], [4, 0, 0, 0]]
STAGE_3 = [[16384, 8192, 1024, 64], [128, 256, 32, 4], [16, 8, 2, 0], [2, 0, 0, 0]]
STAGE_4 = [[32768, 4096, 1024, 16], [512, 256, 64, 4], [8, 32, 2, 0], [4, 0, 0, 0]]
STAGE_5 = [[32768, 8192, 2048, 128], [256, 512, 64, 8], [16, 32, 4, 0], [2, 0, 0, 0]]


class GpuParity(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def test_moves_rewards_and_tuple_values(self):
        rng = np.random.default_rng(20260926)
        ranks = rng.integers(0, 18, (600, 4, 4), dtype=np.uint32)
        ranks[rng.random(ranks.shape) < 0.4] = 0
        grids = np.where(ranks, 1 << ranks, 0).tolist()
        grids += [[[0]*4 for _ in range(4)], [[2]*4 for _ in range(4)], [[32768]*4 for _ in range(4)], STAGE_2, STAGE_3,
                  STAGE_4, STAGE_5, [row[::-1] for row in STAGE_3[::-1]], [row[::-1] for row in STAGE_5],
                  [[0, 0, 0, 0], [0, 16384, 2, 0], [0, 0, 0, 0], [0, 0, 8192, 4]],
                  [[65536, 32768, 0, 0], [0, 16384, 2, 0], [0, 0, 0, 0], [0, 0, 8192, 4]]]
        for layout in module.LAYOUTS:
            with self.subTest(layout=layout):
                self.check_boards(grids, layout)

    def check_boards(self, grids, layout):
        reference = cpu_reference({"mode": "boards", "boards": grids, "layout": layout})
        gpu = module.GpuTrainer(games=1, tc=False, capacity=16, layout=layout)
        # A deterministic nonuniform table exposes wrong pattern/symmetry/stage order.
        for start in range(0, gpu.nw, 1 << 22):
            gpu.w[start:start+(1 << 22)] = (cp.arange(start, start+(1 << 22), dtype=cp.int32) % 1024) / 128
        boards = cp.asarray(np.array([rows(grid) for grid in grids], dtype=np.uint32))
        out = cp.empty((len(grids), 4, 6), dtype=cp.uint32)
        values = cp.empty(len(grids), dtype=cp.float64)
        gpu.module.get_function("inspect_boards")(((len(grids)+127)//128,), (128,),
            (boards, out, values, gpu.w, gpu.left, gpu.right, gpu.rewards, np.int32(len(grids))))
        out, values = out.get(), values.get()
        for i, expected in enumerate(reference):
            self.assertEqual(values[i], expected["value"])
            for direction, move in enumerate(expected["moves"]):
                self.assertEqual(bool(out[i, direction, 5]), move["grid"] is not None)
                self.assertEqual(out[i, direction, 4], move["reward"])
                np.testing.assert_array_equal(out[i, direction, :4], rows(move["grid"] or grids[i]))
        del gpu
        self.tearDown()

    def test_whole_episodes_updates_and_history_growth(self):
        # The exact build must match the CPU bit for bit; the training build's
        # hardware atomics round each increment to float first.
        for tc, exact in ((False, True), (True, True), (False, False), (True, False)):
            with self.subTest(tc=tc, exact=exact):
                check = np.testing.assert_array_equal if exact else \
                    lambda actual, desired: np.testing.assert_allclose(actual, desired, rtol=1e-6)
                expected = cpu_reference({"mode": "episodes", "tc": tc, "seed": 98765})
                gpu = module.GpuTrainer(games=1, tc=tc, seed=98765, alpha=0.3, restart=0, capacity=16, exact=exact)
                episodes = []
                for _ in range(10000):
                    for start, score, tile, steps in gpu.advance(32):
                        episodes.append({"score": int(score), "steps": int(steps), "maxTile": int(tile)})
                    if len(episodes) == 3:
                        break
                self.assertEqual(episodes, expected["episodes"])
                self.assertGreater(gpu.capacity, 16)
                self.check_updates(gpu, expected, check)
                del gpu
                self.tearDown()

    def check_updates(self, gpu, expected, check):
        changed = np.array(expected["changed"], dtype=np.float64)
        indexes = cp.asarray(changed[:, 0].astype(np.int32))
        check(gpu.w[indexes].get(), changed[:, 1])
        if gpu.use_tc:
            check(gpu.e[indexes].get(), changed[:, 2])
            check(gpu.a[indexes].get(), changed[:, 3])
        # Check that no additional weights were changed outside the CPU result.
        actual_changes = 0
        for start in range(0, gpu.nw, 1 << 22):
            initial = initial_weights(cp.arange(start, start+(1 << 22), dtype=cp.int32), gpu.layout)
            actual_changes += int(cp.count_nonzero(gpu.w[start:start+(1 << 22)] != initial).get())
        expected_changes = sum(weight != int(initial_weights(cp.int32(i), gpu.layout))
                               for i, weight, _, _ in expected["changed"])
        self.assertEqual(actual_changes, expected_changes)

    def test_large_network_episodes(self):
        # New games stay in stage 1; restarts from late boards (through their
        # pools) train the later stages.
        late = lambda seed, board: {"seed": seed, "startRows": rows(board), "episodes": 1, "skipRandoms": 3}
        runs = [("large", {"seed": 98765}, None), ("large", late(5, STAGE_2), 1), ("large", late(6, STAGE_3), 2),
                ("xlarge", late(7, STAGE_3), 2), ("xlarge", late(8, STAGE_4), 3), ("xlarge", late(9, STAGE_5), 4)]
        for layout, request, pool in runs:
            with self.subTest(layout=layout, pool=pool):
                expected = cpu_reference({"mode": "episodes", "tc": True, "layout": layout, **request})
                gpu = module.GpuTrainer(games=1, tc=True, seed=request["seed"], alpha=0.3, capacity=16,
                                        restart=0 if pool is None else 0.9, exact=True, layout=layout)
                if pool is not None:
                    gpu.counts[pool, 0] = 1
                    gpu.pools[pool, 0, :, 0] = cp.asarray(request["startRows"], dtype=cp.uint32)
                episodes = []
                for _ in range(10000):
                    for start, score, tile, steps in gpu.advance(32):
                        episodes.append({"score": int(score), "steps": int(steps), "maxTile": int(tile)})
                    if len(episodes) == len(expected["episodes"]):
                        break
                self.assertEqual(episodes, expected["episodes"])
                self.check_updates(gpu, expected, np.testing.assert_array_equal)
                del gpu
                self.tearDown()

    def test_checkpoint_roundtrip(self):
        (ROOT / ".scratch").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            path = Path(directory) / "roundtrip.bin"
            gpu = module.GpuTrainer(games=32, seed=222, capacity=64)
            for _ in range(10):
                gpu.advance(64)
            gpu.save(path, {"games": 10, "steps": 20, "seconds": 1})
            self.assertEqual(path.stat().st_size, module.NW*4)
            self.assertEqual(path.with_suffix(".tc").stat().st_size, module.NW*8)
            for src, offset, target in [(path, 0, gpu.w), (path.with_suffix('.tc'), 0, gpu.e),
                                        (path.with_suffix('.tc'), module.NW*4, gpu.a)]:
                host = np.memmap(src, dtype='<f4', mode='r', offset=offset, shape=(module.NW,))
                np.testing.assert_array_equal(host, target.get())
            restored = module.GpuTrainer(path, games=32, capacity=64)
            self.assertTrue(restored.use_tc)
            self.assertTrue(bool(cp.array_equal(gpu.w, restored.w)))
            self.assertTrue(bool(cp.array_equal(gpu.e, restored.e)))
            self.assertTrue(bool(cp.array_equal(gpu.a, restored.a)))
            # A large network started from it: every stage copies its patterns
            # (the extra four start at 0); only stage 1 keeps its adaptive state.
            large = module.GpuTrainer(Path(directory) / "large.bin", games=32, capacity=64, layout="large", source=path)
            block = module.BLOCK
            for stage in range(3):
                first = stage * 8 * block
                self.assertTrue(bool(cp.array_equal(large.w[first:first + 4*block], gpu.w)))
                self.assertFalse(bool(large.w[first + 4*block:first + 8*block].any()))
                for mine, theirs in ((large.e, gpu.e), (large.a, gpu.a)):
                    part = mine[first:first + 4*block]
                    self.assertTrue(bool(cp.array_equal(part, theirs)) if stage == 0 else not bool(part.any()))
            del large
            self.tearDown()
            # Extra large from large: stages 4-5 copy stages 2-3 (weights and rates).
            self.assertEqual([module.source_stage(s, 3) for s in range(5)], [0, 1, 2, 1, 2])
            self.assertEqual([module.source_stage(s, 1) for s in range(5)], [0, 0, 0, 0, 0])
            # Weight-only checkpoints intentionally retain fixed-rate behavior.
            path.with_suffix('.tc').unlink()
            del gpu, restored
            self.tearDown()
            restored = module.GpuTrainer(path, games=32, capacity=64)
            self.assertFalse(restored.use_tc)
            self.assertEqual(restored.alpha, 0.1)

    def test_late_game_restart(self):
        initial = rows([[8192, 128, 16, 2], [64, 32, 4, 0], [8, 2, 0, 0], [0, 0, 0, 0]])
        expected = cpu_reference({"mode": "episodes", "tc": True, "seed": 1,
                                  "startRows": initial, "episodes": 1, "skipRandoms": 3})
        gpu = module.GpuTrainer(games=1, tc=True, seed=1, alpha=0.3, restart=0.9, capacity=16)
        gpu.counts[0, 0] = 1
        gpu.pools[0, 0, :, 0] = cp.asarray(initial, dtype=cp.uint32)
        for _ in range(10000):
            completed = gpu.advance(32)
            if len(completed):
                break
        self.assertEqual(len(completed), 1)
        start, score, tile, steps = map(int, completed[0])
        self.assertEqual(start, 13)
        self.assertEqual({"score": score, "maxTile": tile, "steps": steps}, expected["episodes"][0])


if __name__ == '__main__':
    unittest.main()
