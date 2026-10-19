"""Recentre colmap2nerf's transforms.json on the scene and write the train/test split.

colmap2nerf.py centres the scene where the cameras' optical axes cross. For a
sideways (forward-facing) capture the axes are almost parallel, so that point
lands far behind the scene. Here the centre is instead taken from the COLMAP
sparse points (median), mapped into the transforms.json frame with a similarity
transform fitted on the camera centres.

Split rule matches 3DGS --eval (scene/dataset_readers.py): sorted image names,
index % llffhold == 0 -> test.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_colmap_centres(images_txt):
    # images.txt: two lines per image; the first has 10 fields ending in the name, the
    # second (2D points) may be empty, so empty lines must be kept to stay paired
    centres = {}
    lines = [l for l in Path(images_txt).read_text().splitlines() if not l.startswith("#")]
    for line in lines[0::2]:
        el = line.split()
        if len(el) < 10:
            continue
        q = np.array(list(map(float, el[1:5])))
        t = np.array(list(map(float, el[5:8])))
        centres[" ".join(el[9:])] = -qvec2rotmat(q).T @ t
    return centres


def read_colmap_points(points_txt):
    pts = [l.split()[1:4] for l in Path(points_txt).read_text().splitlines() if l and not l.startswith("#")]
    return np.array(pts, dtype=float)


def fit_similarity(a, b):
    # Umeyama: find s, R, t with b ~= s * R @ a + t
    ma, mb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((a - ma).T @ (b - mb))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ d @ u.T
    s = (sv * np.diag(d)).sum() / ((a - ma) ** 2).sum()
    return s, r, mb - s * r @ ma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="data/scene")
    p.add_argument("--llffhold", type=int, default=8)
    p.add_argument("--radius", type=float, default=1.0, help="farthest camera distance from the centre after rescaling")
    p.add_argument("--results", default="results")
    args = p.parse_args()

    scene, res = Path(args.scene), Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    data = json.loads((scene / "transforms.json").read_text())
    frames = sorted(data["frames"], key=lambda f: Path(f["file_path"]).name)
    names = [Path(f["file_path"]).name for f in frames]
    mats = np.array([f["transform_matrix"] for f in frames], dtype=float)

    # map COLMAP sparse points into the transforms.json frame
    colmap_c = read_colmap_centres(scene / "colmap_text" / "images.txt")
    a = np.array([colmap_c[n] for n in names])
    s, r, t = fit_similarity(a, mats[:, :3, 3])
    resid = np.abs(s * (r @ a.T).T + t - mats[:, :3, 3]).max()
    assert resid < 1e-4, f"camera fit residual too large: {resid}"
    pts = s * (r @ read_colmap_points(scene / "colmap_text" / "points3D.txt").T).T + t
    centre = np.median(pts, axis=0)

    # recentre + rescale camera positions only (rotations untouched)
    mats[:, :3, 3] -= centre
    scale = args.radius / np.linalg.norm(mats[:, :3, 3], axis=1).max()
    mats[:, :3, 3] *= scale

    fwd = -mats[:, :3, 2]  # NeRF/OpenGL convention: camera looks along -z
    in_front = ((-mats[:, :3, 3]) * fwd).sum(1)
    assert np.all(in_front > 0), "scene centre is behind some camera"

    for f, m in zip(frames, mats):
        f["transform_matrix"] = m.tolist()

    test_idx = {i for i in range(len(frames)) if i % args.llffhold == 0}
    splits = {
        "train": [f for i, f in enumerate(frames) if i not in test_idx],
        "test": [f for i, f in enumerate(frames) if i in test_idx],
    }
    for split, fr in splits.items():
        out = copy.deepcopy({k: v for k, v in data.items() if k != "frames"})
        out["frames"] = fr
        (scene / f"transforms_{split}.json").write_text(json.dumps(out, indent=2))
        (res / f"{split}_list.txt").write_text("\n".join(Path(f["file_path"]).name for f in fr) + "\n")

    print(f"fit residual {resid:.2e}, colmap->nerf scale {s:.4f}")
    print(f"centre moved by {np.linalg.norm(centre):.3f} (old units), rescale x{scale:.3f}")
    print(f"camera distance to centre: min {np.linalg.norm(mats[:, :3, 3], axis=1).min():.3f} max {args.radius:.3f}")
    print(f"points 5-95% after: {((np.percentile(pts, 5, 0) - centre) * scale).round(2)} {((np.percentile(pts, 95, 0) - centre) * scale).round(2)}")
    print(f"train {len(splits['train'])}  test {len(splits['test'])}: {[Path(f['file_path']).name for f in splits['test']]}")


if __name__ == "__main__":
    main()
