"""Render a trained 3DGS model: the test split, a novel camera path, or a timing benchmark.

Run from the assignment root inside the 3DGS conda env, e.g.
  python scripts/render_gs.py -m outputs/gs --iteration 30000 --out outputs/eval/gs/iter_030000
  python scripts/render_gs.py -m outputs/gs --novel data/scene/novel/orbit.json --out outputs/novel/gs/orbit
  python scripts/render_gs.py -m outputs/gs --bench 20 --timing_out results/timing/gs.json

Model loading, cameras and the render call mirror the repo's render.py. Novel views use
the COLMAP-world poses ("colmap_c2w") written by make_novel_paths.py, turned into the same
camera objects 3DGS builds for its own views (scene/cameras.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--gs_root", default="repos/gaussian-splatting")
sys.path.insert(0, str(Path(pre.parse_known_args()[0].gs_root).resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402

import poses  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def camera_from_colmap(c2w, ref):
    """MiniCam at an OpenCV c2w (COLMAP world) with the intrinsics of camera `ref`."""
    rot = c2w[:3, :3]                       # 3DGS stores R = c2w rotation (dataset_readers.py)
    t = -rot.T @ c2w[:3, 3]                 # and T = w2c translation
    wvt = torch.tensor(getWorld2View2(rot, t)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=ref.znear, zfar=ref.zfar, fovX=ref.FoVx, fovY=ref.FoVy).transpose(0, 1).cuda()
    full = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(ref.image_width, ref.image_height, ref.FoVy, ref.FoVx, ref.znear, ref.zfar, wvt, full)


def check_pose_conversion(scene, colmap_text):
    """Rebuild every test camera from COLMAP text poses; must match 3DGS's own camera."""
    _, colmap_c2w, _, _ = poses.read_colmap_text(colmap_text)
    worst = 0.0
    for view in scene.getTestCameras():
        cam = camera_from_colmap(colmap_c2w[view.image_name], view)
        worst = max(worst, float((cam.full_proj_transform - view.full_proj_transform).abs().max()))
    print(f"pose check: max |projection difference| over test cameras = {worst:.2e}")
    if worst > 1e-4:
        raise RuntimeError("novel-view cameras would not match 3DGS's camera convention")


def main():
    parser = argparse.ArgumentParser(description="3DGS renderer for the assignment")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--gs_root", default="repos/gaussian-splatting")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--novel", default="", help="novel path json from make_novel_paths.py")
    parser.add_argument("--colmap_text", default="data/scene/colmap_text")
    parser.add_argument("--out", default="")
    parser.add_argument("--bench", type=int, default=0, help="timed passes over the test views (no images written)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timing_out", default="")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        test_views = scene.getTestCameras()
        if not test_views:
            raise RuntimeError("no test cameras: the model must be trained with --eval")
        print(f"loaded iteration {scene.loaded_iter}: {gaussians.get_xyz.shape[0]} Gaussians, {len(test_views)} test views")

        def draw(view):
            # per-image exposure (train_test_exp) only exists for dataset cameras, not novel ones
            use_exp = dataset.train_test_exp and hasattr(view, "image_name")
            img = render(view, gaussians, pipe, background, use_trained_exp=use_exp,
                         separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            if use_exp:
                img = img[..., img.shape[-1] // 2:]
            return img

        if args.bench > 0:
            for _ in range(args.warmup):
                draw(test_views[0])
            torch.cuda.synchronize()
            times = []
            for _ in range(args.bench):
                for view in test_views:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    draw(view)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000.0)
            times = np.array(times)
            result = {
                "method": "3dgs", "iteration": scene.loaded_iter, "n_gaussians": int(gaussians.get_xyz.shape[0]),
                "width": test_views[0].image_width, "height": test_views[0].image_height,
                "n_views": len(test_views), "repeats": args.bench, "warmup": args.warmup,
                "gpu": torch.cuda.get_device_name(0),
                "ms_mean": float(times.mean()), "ms_std": float(times.std()), "ms_median": float(np.median(times)),
                "fps_mean": float(1000.0 / times.mean()),
            }
            print(json.dumps(result, indent=2))
            if args.timing_out:
                Path(args.timing_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.timing_out).write_text(json.dumps(result, indent=2))
            return

        if not args.out:
            raise SystemExit("--out is required unless --bench is given")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        if args.novel:
            check_pose_conversion(scene, args.colmap_text)
            frames = json.loads(Path(args.novel).read_text())["frames"]
            for i, f in enumerate(frames):
                cam = camera_from_colmap(np.array(f["colmap_c2w"], dtype=np.float64), test_views[0])
                torchvision.utils.save_image(draw(cam), str(out / f"{i:04d}.png"))
            print(f"rendered {len(frames)} novel views -> {out}")
        else:
            for view in test_views:
                torchvision.utils.save_image(draw(view), str(out / f"{Path(view.image_name).stem}.png"))
            print(f"rendered {len(test_views)} test views -> {out}")


if __name__ == "__main__":
    main()
