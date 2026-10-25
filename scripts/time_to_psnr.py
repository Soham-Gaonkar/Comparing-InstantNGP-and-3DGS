"""Training time each method needs to reach a given test PSNR (markdown table).

Times are interpolated between the saved checkpoints in results/summary.json, assuming PSNR is
linear in log(wall-clock time) between two checkpoints. With 5-6 checkpoints per method this is
an estimate, not a measurement.

  python scripts/time_to_psnr.py [--summary results/summary.json] [--targets 28 30 31 32 33 35 36]
"""
import argparse
import json
import math
from pathlib import Path


def time_to(conv, target):
    prev = None
    for c in conv:
        if c["psnr"] >= target:
            if prev is None:
                return c["wall_s"], True          # already reached at the first checkpoint
            f = (target - prev["psnr"]) / (c["psnr"] - prev["psnr"])
            return math.exp(math.log(prev["wall_s"]) + f * (math.log(c["wall_s"]) - math.log(prev["wall_s"]))), False
        prev = c
    return None, False


def fmt(t, upper):
    if t is None:
        return "not reached"
    s = f"{t:.0f} s" if t < 120 else f"{t / 60:.1f} min"
    return f"≤ {s}" if upper else s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.json")
    p.add_argument("--targets", type=float, nargs="+", default=[28, 30, 31, 32, 33, 35, 36])
    a = p.parse_args()

    s = json.loads(Path(a.summary).read_text())
    convs = {m: sorted(s[m]["convergence"], key=lambda c: c["iteration"]) for m in ("instant-ngp", "3dgs")}
    print("| Test PSNR reached | InstantNGP | 3DGS |")
    print("|---|---|---|")
    for target in a.targets:
        t = {m: time_to(c, target) for m, c in convs.items()}
        cells = {m: fmt(*v) for m, v in t.items()}
        known = {m: v[0] for m, v in t.items() if v[0] is not None}
        if len(known) == 2:                       # bold the faster method
            fast = min(known, key=known.get)
            cells[fast] = f"**{cells[fast]}**"
        print(f"| {target:g} dB | {cells['instant-ngp']} | {cells['3dgs']} |")
    best = {m: max(c, key=lambda c: c["psnr"]) for m, c in convs.items()}
    print(f"| Best reached | {best['instant-ngp']['psnr']:.2f} dB at {fmt(best['instant-ngp']['wall_s'], False)} "
          f"| {best['3dgs']['psnr']:.2f} dB at {fmt(best['3dgs']['wall_s'], False)} |")


if __name__ == "__main__":
    main()
