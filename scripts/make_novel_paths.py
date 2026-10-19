"""Generate novel-view camera paths shared by 3DGS and InstantNGP.

Each path is written once to data/scene/novel/<name>.json in InstantNGP's format. Every
frame stores both poses:
  transform_matrix : OpenGL c2w in the transforms_train.json world (InstantNGP)
  colmap_c2w       : OpenCV c2w in the COLMAP world (3DGS)
so both models render exactly the same viewpoints.

Paths (sizes relative to the median scene depth D):
  trajectory   smoothed version of the captured camera path (interpolation)
  shift_up     trajectory moved up by 0.15 D        (vertical extrapolation)
  shift_down   trajectory moved down by 0.15 D      (vertical extrapolation)
  spiral       trajectory with a circular 0.08 D offset, 3 turns (forward-facing "LLFF spiral")
  dolly        mid-path camera moving +-0.35 D towards / away from the shelf
  orbit        arc of +-30 deg around the scene centre, always looking at it
  look_around  mid-path camera panning +-20 deg and tilting +-12 deg
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import poses


def smooth_trajectory(c2ws, frame, n):
    """Kernel-smoothed camera path over the captured cameras, sampled at n points."""
    u = (c2ws[:, :3, 3] - frame["mean_cam"]) @ frame["ex"]
    order = np.argsort(u)
    u, c2ws = u[order], c2ws[order]
    bw = 1.5 * np.median(np.diff(u))
    out = []
    for uq in np.linspace(u[0], u[-1], n):
        w = np.exp(-0.5 * ((u - uq) / bw) ** 2)
        w /= w.sum()
        m = np.eye(4)
        m[:3, :3] = poses.project_to_so3(np.einsum("i,ijk->jk", w, c2ws[:, :3, :3]))
        m[:3, 3] = w @ c2ws[:, :3, 3]
        out.append(m)
    return np.array(out)


def build_paths(c2ws, frame):
    d, ex, ey, ez = frame["depth"], frame["ex"], frame["ey"], frame["ez"]
    traj = smooth_trajectory(c2ws, frame, 120)
    mid = traj[len(traj) // 2]

    def shifted(base, offsets):
        out = base.copy()
        out[:, :3, 3] += offsets
        return out

    sub = smooth_trajectory(c2ws, frame, 90)
    t90 = np.linspace(0, 1, 90, endpoint=False)
    t120 = np.linspace(0, 1, 120, endpoint=False)

    paths = {}
    paths["trajectory"] = (traj, "smoothed captured path (interpolation)")
    paths["shift_up"] = (shifted(sub, -0.15 * d * ey), "captured path moved up by 0.15 x scene depth")
    paths["shift_down"] = (shifted(sub, 0.15 * d * ey), "captured path moved down by 0.15 x scene depth")
    spiral_off = 0.08 * d * (np.cos(6 * np.pi * t120)[:, None] * ex + np.sin(6 * np.pi * t120)[:, None] * ey)
    paths["spiral"] = (shifted(traj, spiral_off), "captured path + circular offset of 0.08 x scene depth")
    paths["dolly"] = (shifted(np.repeat(mid[None], 90, 0), (0.35 * d * np.sin(2 * np.pi * t90))[:, None] * ez),
                      "mid-path camera moving +-0.35 x scene depth along the viewing axis")

    # orbit: keep the mean camera height and distance, swing +-30 deg around the scene centre
    height = (frame["mean_cam"] - frame["centre"]) @ ey
    radius = (frame["centre"] - frame["mean_cam"]) @ ez
    orbit = []
    for phi in np.radians(30) * np.sin(2 * np.pi * t120):
        pos = frame["centre"] + height * ey + radius * (np.sin(phi) * ex - np.cos(phi) * ez)
        orbit.append(poses.look_at(pos, frame["centre"] + height * ey, ey))
    paths["orbit"] = (np.array(orbit), "+-30 deg arc around the scene centre, looking at it")

    look = []
    for t in t90:
        yaw, pitch = np.radians(20) * np.sin(2 * np.pi * t), np.radians(12) * np.sin(4 * np.pi * t)
        m = mid.copy()
        m[:3, :3] = mid[:3, :3] @ poses.axis_angle([0, 1, 0], yaw) @ poses.axis_angle([1, 0, 0], pitch)
        look.append(m)
    paths["look_around"] = (np.array(look), "mid-path camera panning +-20 deg / tilting +-12 deg")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="data/scene")
    p.add_argument("--out", default="data/scene/novel")
    p.add_argument("--preview", default="results/figures/novel_paths_preview.png")
    args = p.parse_args()

    scene, out_dir = Path(args.scene), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam, colmap_c2w, pts, _ = poses.read_colmap_text(scene / "colmap_text")
    train, train_m = poses.load_ngp_json(scene / "transforms_train.json")
    _, test_m = poses.load_ngp_json(scene / "transforms_test.json")
    ngp_c2w = {**train_m, **test_m}

    fit = poses.fit_world_transform(colmap_c2w, ngp_c2w)
    print(f"world fit on {len(fit['names'])} cameras: scale {fit['s']:.5f}, "
          f"max position error {fit['max_pos_err']:.2e}, max rotation error {fit['max_rot_err_deg']:.2e} deg")
    assert fit["max_pos_err"] < 1e-5 * max(1.0, fit["scene_scale"]) and fit["max_rot_err_deg"] < 1e-3, "COLMAP and transforms poses disagree"

    # intrinsics must agree between COLMAP and transforms_train.json
    for k_json, k_cam in [("fl_x", "fx"), ("fl_y", "fy"), ("cx", "cx"), ("cy", "cy"), ("w", "w"), ("h", "h")]:
        assert abs(float(train[k_json]) - float(cam[k_cam])) < 1e-6, f"{k_json} differs from COLMAP"

    names = sorted(colmap_c2w)
    c2ws = np.array([colmap_c2w[n] for n in names])
    frame = poses.scene_frame(c2ws, pts)
    print(f"scene depth D = {frame['depth']:.3f}, camera path length L = {frame['length']:.3f} (COLMAP units)")

    pts_ngp = poses.points_colmap_to_ngp(pts, fit)
    template = {k: v for k, v in train.items() if k != "frames"}
    first_image = "../" + Path(sorted(train["frames"], key=lambda f: f["file_path"])[0]["file_path"]).as_posix().lstrip("./")
    paths = build_paths(c2ws, frame)

    print(f"{'path':12s} {'frames':>6s} {'max reproj diff px':>19s} {'visible pts min/mean':>21s}")
    summary = {}
    for name, (mats, desc) in paths.items():
        frames, max_diff, vis = [], 0.0, []
        for m in mats:
            m_ngp = poses.colmap_to_ngp(m, fit)
            uv_cv, z_cv = poses.project_opencv(pts, m, cam)
            uv_gl, z_gl = poses.project_opengl(pts_ngp, m_ngp, cam)
            ok = z_cv > 1e-6
            max_diff = max(max_diff, float(np.abs(uv_cv[ok] - uv_gl[ok]).max()))
            vis.append(poses.visible_fraction(uv_cv, z_cv, cam))
            frames.append({"file_path": first_image, "transform_matrix": m_ngp.tolist(), "colmap_c2w": m.tolist()})
        assert max_diff < 1e-3, f"{name}: 3DGS and InstantNGP cameras disagree by {max_diff} px"
        assert min(vis) > 0.05, f"{name}: a frame sees almost none of the scene"
        doc = copy.deepcopy(template)
        doc.update({"path": name, "description": desc, "frames": frames})
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=1))
        summary[name] = dict(frames=len(frames), description=desc, min_visible=min(vis), mean_visible=float(np.mean(vis)))
        print(f"{name:12s} {len(frames):6d} {max_diff:19.2e} {min(vis):10.2f} / {np.mean(vis):.2f}")

    (out_dir / "paths_summary.json").write_text(json.dumps(summary, indent=2))

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def coords(x):
            c = x - frame["centre"]
            return c @ frame["ex"], -(c @ frame["ey"]), c @ frame["ez"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        px, py, pz = coords(pts)
        keep = (np.abs(pz) < 0.5 * frame["depth"]) & (np.abs(py) < 0.5 * frame["depth"])
        cx, cy, cz = coords(c2ws[:, :3, 3])
        for ax, (a, b, la, lb) in zip(axes, [(px, pz, "along path", "depth (into scene)"), (px, py, "along path", "height")]):
            ax.scatter(a[keep], b[keep], s=0.3, c="0.75", label="COLMAP points")
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
            ax.set_aspect("equal")
        axes[0].scatter(cx, cz, s=14, c="k", marker="^", label="captured cameras", zorder=3)
        axes[1].scatter(cx, cy, s=14, c="k", marker="^", label="captured cameras", zorder=3)
        colors = plt.cm.tab10(np.arange(len(paths)))
        for col, (name, (mats, _)) in zip(colors, paths.items()):
            x, y, z = coords(mats[:, :3, 3])
            axes[0].plot(x, z, c=col, lw=1.4, label=name)
            axes[1].plot(x, y, c=col, lw=1.4)
        axes[0].set_title("Top view")
        axes[1].set_title("Front view")
        axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=5, frameon=False)
        fig.tight_layout()
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=130, bbox_inches="tight")
        print(f"preview -> {args.preview}")


if __name__ == "__main__":
    main()
