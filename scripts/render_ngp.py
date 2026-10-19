"""Render a trained InstantNGP snapshot: the test split, a novel camera path, or a timing benchmark.

Run from the assignment root inside the InstantNGP env, e.g.
  python scripts/render_ngp.py --snapshot outputs/ngp/step_035000.ingp \
      --transforms data/scene/transforms_test.json --out outputs/eval/ngp/step_035000
  python scripts/render_ngp.py --snapshot outputs/ngp/step_035000.ingp \
      --transforms data/scene/novel/orbit.json --out outputs/novel/ngp/orbit
  python scripts/render_ngp.py --snapshot outputs/ngp/step_035000.ingp \
      --transforms data/scene/transforms_test.json --bench 20 --timing_out results/timing/ngp.json

Rendering follows the test evaluation in instant-ngp/scripts/run.py: cameras come from a
dataset json via set_camera_to_training_view (exact fl_x/fl_y/cx/cy), black background,
snap_to_pixel_centers, render_min_transmittance 1e-4, linear -> sRGB.

InstantNGP's loader sorts frames by file_path and skips frames whose image is missing, so
each chunk of cameras is staged as a temporary json whose frames point to sequentially
named copies of a real image. The image content is never used, only its resolution.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def stage_chunk(frames, template_img, src_json, stage_dir, idx0):
    """Write <stage_dir>.json with frames pointing to stage_dir/00000.jpg, 00001.jpg, ..."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for k, f in enumerate(frames):
        img = stage_dir / f"{idx0 + k:06d}{template_img.suffix}"
        if not img.exists():
            try:
                os.link(template_img, img)
            except OSError:
                shutil.copyfile(template_img, img)
        staged.append({"file_path": f"{stage_dir.name}/{img.name}", "transform_matrix": f["transform_matrix"]})
    doc = {k: v for k, v in src_json.items() if k not in ("frames", "path", "description")}
    doc["frames"] = staged
    path = stage_dir.parent / f"{stage_dir.name}.json"
    path.write_text(json.dumps(doc))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--transforms", required=True, help="transforms_test.json or a novel path json")
    p.add_argument("--train_json", default="data/scene/transforms_train.json", help="for the aabb_scale consistency check")
    p.add_argument("--out", default="")
    p.add_argument("--spp", type=int, default=8, help="samples per pixel for saved images (run.py uses 8)")
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--bench", type=int, default=0, help="timed passes over the views (no images written)")
    p.add_argument("--bench_spp", type=int, nargs="+", default=[1, 8])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timing_out", default="")
    p.add_argument("--ngp_root", default="repos/instant-ngp")
    args = p.parse_args()

    sys.path.insert(0, str((Path(args.ngp_root) / "scripts").resolve()))
    from common import ROOT_DIR, linear_to_srgb  # also puts the pyngp build folder on sys.path
    import pyngp as ngp

    src_path = Path(args.transforms).resolve()
    src = json.loads(src_path.read_text())
    train = json.loads(Path(args.train_json).read_text())
    for key in ("aabb_scale", "scale", "offset"):
        if src.get(key) != train.get(key):
            raise ValueError(f"{key} differs from the training json ({src.get(key)} vs {train.get(key)}); "
                             "InstantNGP would place or reset the model differently")
    w, h = int(src["w"]), int(src["h"])
    is_novel = "path" in src
    frames = src["frames"] if is_novel else sorted(src["frames"], key=lambda f: Path(f["file_path"]).name)
    names = [f"{i:04d}" for i in range(len(frames))] if is_novel else [Path(f["file_path"]).stem for f in frames]

    template = (src_path.parent / frames[0]["file_path"]).resolve()
    if not template.exists():
        raise FileNotFoundError(f"template image {template} not found")
    with Image.open(template) as im:
        if im.size != (w, h):
            raise ValueError(f"template image is {im.size}, json says {(w, h)}")

    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR
    testbed.load_snapshot(str(Path(args.snapshot).resolve()))
    testbed.shall_train = False
    testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    testbed.snap_to_pixel_centers = True
    testbed.nerf.render_min_transmittance = 1e-4
    print(f"snapshot {args.snapshot}: training step {testbed.training_step}, {len(frames)} views at {w}x{h}")

    stage_dir = src_path.parent / f"_ngp_stage_{os.getpid()}"
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    times = {s: [] for s in args.bench_spp}
    try:
        chunk = len(frames) if args.bench else args.chunk
        for c0 in range(0, len(frames), chunk):
            part = frames[c0:c0 + chunk]
            staged = stage_chunk(part, template, src, stage_dir, c0)
            testbed.load_training_data(str(staged))
            n = testbed.nerf.training.dataset.n_images
            if n != len(part):
                raise RuntimeError(f"InstantNGP loaded {n} of {len(part)} staged views")

            if args.bench:
                for spp in args.bench_spp:
                    testbed.set_camera_to_training_view(0)
                    for _ in range(args.warmup):
                        testbed.render(w, h, spp, True)
                    for _ in range(args.bench):
                        for i in range(n):
                            testbed.set_camera_to_training_view(i)
                            t0 = time.perf_counter()
                            testbed.render(w, h, spp, True)   # returns a host copy, so the GPU work is finished
                            times[spp].append((time.perf_counter() - t0) * 1000.0)
                continue

            for i in range(n):
                testbed.set_camera_to_training_view(i)
                img = testbed.render(w, h, args.spp, True)
                rgb = np.clip(linear_to_srgb(img[..., :3]), 0.0, 1.0)
                Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(out / f"{names[c0 + i]}.png")
            print(f"rendered {min(c0 + chunk, len(frames))}/{len(frames)}", flush=True)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        (stage_dir.parent / f"{stage_dir.name}.json").unlink(missing_ok=True)

    if args.bench:
        result = {"method": "instant-ngp", "snapshot": str(args.snapshot), "training_step": int(testbed.training_step),
                  "width": w, "height": h, "n_views": len(frames), "repeats": args.bench, "warmup": args.warmup,
                  "n_params": int(testbed.n_params()), "spp": {}}
        try:
            import subprocess
            result["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip().splitlines()[0]
        except Exception:
            result["gpu"] = "unknown"
        for spp, t in times.items():
            t = np.array(t)
            result["spp"][str(spp)] = {"ms_mean": float(t.mean()), "ms_std": float(t.std()),
                                       "ms_median": float(np.median(t)), "fps_mean": float(1000.0 / t.mean())}
        print(json.dumps(result, indent=2))
        if args.timing_out:
            Path(args.timing_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.timing_out).write_text(json.dumps(result, indent=2))
    elif out:
        print(f"done -> {out}")


if __name__ == "__main__":
    main()
