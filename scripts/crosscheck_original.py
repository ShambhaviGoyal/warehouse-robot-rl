"""Replays a fixed action sequence through the ORIGINAL Python environment (read-only)
and dumps observations/rewards to JSON, so the TypeScript port can be compared to it.
Usage: python3 crosscheck_original.py <path-to-a1_part_1.ipynb> <out.json>
gymnasium/matplotlib are stubbed because only the step/reset logic is exercised."""
import json, sys, types
import numpy as np

class _Env:
    def __init__(self): pass
    def reset(self, seed=None, options=None):
        self.np_random = np.random.default_rng(seed)
class _Discrete:
    def __init__(self, n): self.n = n
class _MultiDiscrete:
    def __init__(self, v): self.v = v
gym = types.ModuleType("gym"); gym.Env = _Env
spaces = types.ModuleType("spaces"); spaces.Discrete = _Discrete; spaces.MultiDiscrete = _MultiDiscrete
from pathlib import Path
nb = json.load(open(sys.argv[1]))
src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "class WarehouseRobotEnv" in "".join(c["source"]))
ns = {"gym": gym, "spaces": spaces, "np": np, "Path": Path, "plt": None}
exec(src, ns)
Env = ns["WarehouseRobotEnv"]

rng = np.random.default_rng(42)
out = []
for start in ([0, 0], [1, 2], [0, 4], [5, 5], [4, 1], [3, 4]):
    env = Env(is_stochastic=False)
    env.reset(seed=0)
    env.robot_position_X, env.robot_position_Y = start
    actions = [int(a) for a in rng.integers(0, 6, size=150)]
    if start == [0, 4]:      # full delivery: pick, walk to dock, drop
        actions = [5, 4, 4, 3, 3, 3, 3, 3, 1, 5]
    if start == [1, 2]:      # shelf bump, wall bumps, pickup away from item
        actions = [3, 3, 0, 0, 0, 2, 2, 4, 5, 1, 1, 1, 1, 1, 1]
    steps = []
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        steps.append([int(o) for o in obs] + [float(r), bool(term), bool(trunc)])
        if term or trunc: break
    out.append({"start": start, "actions": actions[:len(steps)], "steps": steps})
json.dump(out, open(sys.argv[2], "w"))
print("recorded", sum(len(o["steps"]) for o in out), "steps")
