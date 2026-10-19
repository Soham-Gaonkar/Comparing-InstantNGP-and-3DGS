"""Train InstantNGP in one run and save snapshots at fixed steps (for the convergence curve).

Mirrors the training loop of instant-ngp/scripts/run.py (same Testbed setup, default
network config, one training step per testbed.frame()). Snapshot saving is excluded from
the timed training time; everything is recorded in <out_dir>/train_log.json.

  python scripts/train_ngp.py --scene data/scene/transforms_train.json --out_dir outputs/ngp \
      --checkpoints 1000 3000 7000 15000 35000
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="data/scene/transforms_train.json")
    p.add_argument("--out_dir", default="outputs/ngp")
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 3000, 7000, 15000, 35000])
    p.add_argument("--network", default="", help="network config; empty = instant-ngp default (configs/nerf/base.json)")
    p.add_argument("--ngp_root", default="repos/instant-ngp")
    args = p.parse_args()

    sys.path.insert(0, str((Path(args.ngp_root) / "scripts").resolve()))
    from common import ROOT_DIR  # also puts the pyngp build folder on sys.path
    import pyngp as ngp

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(set(args.checkpoints))

    t_start = time.perf_counter()
    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR
    testbed.load_training_data(str(Path(args.scene).resolve()))
    if args.network:
        testbed.reload_network_from_file(args.network)
    testbed.shall_train = True
    load_s = time.perf_counter() - t_start

    log = {"scene": str(args.scene), "network": args.network or "default (configs/nerf/base.json)",
           "load_s": load_s, "checkpoints": []}
    train_s = 0.0
    seg_start = time.perf_counter()
    next_print = 1000
    while testbed.frame():
        step = testbed.training_step
        if step >= next_print:
            print(f"step {step:6d}  loss {testbed.loss:.5f}  train time {train_s + time.perf_counter() - seg_start:7.1f} s", flush=True)
            next_print += 1000
        if step >= checkpoints[0]:
            train_s += time.perf_counter() - seg_start
            path = out / f"step_{checkpoints[0]:06d}.ingp"
            testbed.save_snapshot(str(path), False)
            log["checkpoints"].append({"target_step": checkpoints[0], "step": step, "train_s": train_s,
                                       "wall_s": time.perf_counter() - t_start,
                                       "loss": float(testbed.loss), "snapshot": str(path),
                                       "snapshot_bytes": path.stat().st_size})
            print(f"checkpoint {path}  ({train_s:.1f} s of training)", flush=True)
            checkpoints.pop(0)
            if not checkpoints:
                break
            seg_start = time.perf_counter()

    if checkpoints:
        raise RuntimeError(f"training stopped before reaching steps {checkpoints}")

    # uncompressed copy of the final model, to report the raw parameter size too
    raw = out / "final_uncompressed.msgpack"
    testbed.save_snapshot(str(raw), False)
    log["final_uncompressed_bytes"] = raw.stat().st_size
    log["n_params"] = int(testbed.n_params())
    log["n_encoding_params"] = int(testbed.n_encoding_params())
    log["train_s"] = train_s
    log["total_wall_s"] = time.perf_counter() - t_start
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
