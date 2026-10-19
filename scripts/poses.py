"""Camera-pose utilities shared by the novel-view, rendering and figure scripts.

Two coordinate systems are in play:

* COLMAP / 3DGS: OpenCV cameras (x right, y down, z forward) in COLMAP's world.
* transforms*.json (InstantNGP): OpenGL cameras (x right, y up, z backward) in the
  world written by colmap2nerf.py and recentred by prepare_ngp.py.

The worlds differ by a similarity transform (rotation R, translation t, scale s) and the
camera axes differ by F = diag(1, -1, -1):

    rot_ngp    = R @ rot_colmap @ F
    centre_ngp = s * R @ centre_colmap + t

fit_world_transform() recovers (s, R, t) from cameras present in both files.
"""
import json
from pathlib import Path

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def project_to_so3(m):
    u, _, vt = np.linalg.svd(m)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ d @ vt


def axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def read_colmap_text(text_dir):
    """Return (camera, {name: c2w OpenCV 4x4}, points (N,3), colors (N,3) in [0,1])."""
    text_dir = Path(text_dir)
    cams = []
    for line in (text_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        el = line.split()
        model, w, h, p = el[1], int(el[2]), int(el[3]), list(map(float, el[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = p
        elif model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            raise ValueError(f"expected an undistorted PINHOLE camera, got {model}")
        cams.append(dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy))
    if len(cams) != 1:
        raise ValueError(f"expected exactly one camera, found {len(cams)}")

    # images.txt: two lines per image; the second (2D points) may be empty, so keep empty lines
    lines = [l for l in (text_dir / "images.txt").read_text().splitlines() if not l.startswith("#")]
    c2w = {}
    for line in lines[0::2]:
        el = line.split()
        if len(el) < 10:
            continue
        rot = qvec2rotmat(np.array(list(map(float, el[1:5]))))
        t = np.array(list(map(float, el[5:8])))
        m = np.eye(4)
        m[:3, :3] = rot.T
        m[:3, 3] = -rot.T @ t
        c2w[" ".join(el[9:])] = m

    pts, cols = [], []
    for line in (text_dir / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        el = line.split()
        pts.append(el[1:4])
        cols.append(el[4:7])
    return cams[0], c2w, np.array(pts, float), np.array(cols, float) / 255.0


def load_ngp_json(path):
    """Return (json dict, {image name: c2w OpenGL 4x4})."""
    data = json.loads(Path(path).read_text())
    mats = {Path(f["file_path"]).name: np.array(f["transform_matrix"], float) for f in data["frames"]}
    return data, mats


def fit_world_transform(colmap_c2w, ngp_c2w):
    """Fit s, R, t with ngp = similarity(colmap) on the cameras both dicts share."""
    names = sorted(set(colmap_c2w) & set(ngp_c2w))
    if len(names) < 3:
        raise ValueError("need at least 3 shared cameras to fit the world transform")
    rc = np.array([colmap_c2w[n][:3, :3] for n in names])
    rn = np.array([ngp_c2w[n][:3, :3] for n in names])
    rot = project_to_so3(sum(b @ FLIP @ a.T for a, b in zip(rc, rn)))

    pc = np.array([colmap_c2w[n][:3, 3] for n in names])
    pn = np.array([ngp_c2w[n][:3, 3] for n in names])
    mc, mn = pc.mean(0), pn.mean(0)
    s = float(((rot @ (pc - mc).T).T * (pn - mn)).sum() / ((pc - mc) ** 2).sum())
    t = mn - s * rot @ mc

    fit = dict(s=s, R=rot, t=t, names=names)
    pos_err = np.linalg.norm(np.array([colmap_to_ngp(colmap_c2w[n], fit)[:3, 3] for n in names]) - pn, axis=1)
    rot_err = [np.degrees(np.arccos(np.clip((np.trace(colmap_to_ngp(colmap_c2w[n], fit)[:3, :3].T @ ngp_c2w[n][:3, :3]) - 1) / 2, -1, 1)))
               for n in names]
    fit["max_pos_err"] = float(pos_err.max())
    fit["max_rot_err_deg"] = float(max(rot_err))
    fit["scene_scale"] = float(np.linalg.norm(pn - mn, axis=1).max())
    return fit


def colmap_to_ngp(c2w_cv, fit):
    out = np.eye(4)
    out[:3, :3] = fit["R"] @ c2w_cv[:3, :3] @ FLIP
    out[:3, 3] = fit["s"] * fit["R"] @ c2w_cv[:3, 3] + fit["t"]
    return out


def ngp_to_colmap(c2w_gl, fit):
    out = np.eye(4)
    out[:3, :3] = fit["R"].T @ c2w_gl[:3, :3] @ FLIP
    out[:3, 3] = fit["R"].T @ (c2w_gl[:3, 3] - fit["t"]) / fit["s"]
    return out


def points_colmap_to_ngp(x, fit):
    return fit["s"] * (fit["R"] @ x.T).T + fit["t"]


def project_opencv(x, c2w, cam):
    """Pixel coords (N,2) and depth (N,) of world points for an OpenCV camera."""
    xc = (c2w[:3, :3].T @ (x - c2w[:3, 3]).T).T
    z = xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([cam["fx"] * xc[:, 0] / z + cam["cx"], cam["fy"] * xc[:, 1] / z + cam["cy"]], 1)
    return uv, z


def project_opengl(x, c2w, cam):
    """Same as project_opencv for an OpenGL camera (y up, looking down -z)."""
    xc = (c2w[:3, :3].T @ (x - c2w[:3, 3]).T).T
    z = -xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([cam["fx"] * xc[:, 0] / z + cam["cx"], -cam["fy"] * xc[:, 1] / z + cam["cy"]], 1)
    return uv, z


def visible_fraction(uv, z, cam):
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"])
    return float(ok.mean())


def scene_frame(c2ws, points):
    """Axes of the capture in COLMAP world: ex along the camera path, ey down, ez into the scene."""
    centres = c2ws[:, :3, 3]
    ex = np.linalg.svd(centres - centres.mean(0))[2][0]
    if ex @ c2ws[:, :3, 0].mean(0) < 0:
        ex = -ex
    down = c2ws[:, :3, 1].mean(0)
    ey = down - (down @ ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    if ez @ c2ws[:, :3, 2].mean(0) < 0:
        raise ValueError("camera axes are inconsistent with a right-handed OpenCV frame")
    centre = np.median(points, 0)
    u = (centres - centres.mean(0)) @ ex
    return dict(
        ex=ex, ey=ey, ez=ez, centre=centre, mean_cam=centres.mean(0),
        depth=float(np.median((centre - centres) @ ez)),
        length=float(u.max() - u.min()),
    )


def look_at(pos, target, down):
    """OpenCV c2w at pos looking at target, with image 'down' close to the given vector."""
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z], 1)
    m[:3, 3] = pos
    return m
