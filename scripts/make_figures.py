"""All report figures and side-by-side videos for InstantNGP vs 3DGS.

  python scripts/make_figures.py            (after collect_results.py)

Figures (results/figures/):
  01_test_views          largest-gap and worst test view: GT | InstantNGP | 3DGS | error maps
  02_zoom_crops          high-detail crops of the hardest / most different views
  04_per_view_metrics    PSNR / SSIM / LPIPS per test view
  05_convergence         metrics vs iterations and vs training wall-clock time
  06_efficiency          training time, render speed, model size, GPU memory
  07_model_growth        model size over training
  08_error_distribution  per-pixel error histogram and CDF
  09_view_extrapolation  PSNR vs distance of a test view to the nearest training camera
  10_sharpness           render sharpness relative to GT per test view
  11_frequency_spectrum  radial power spectrum of renders relative to GT
  12_novel_overview      the most extreme frame of every novel path, both methods
  13_novel_<path>        frame strips along each novel path
  14_camera_paths        captured, test and novel cameras over the COLMAP points
  15_gaussian_stats      3DGS opacity / size distributions and count over training
Videos (results/videos/): <path>_side_by_side.mp4 for every novel path.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.ndimage import gaussian_filter, laplace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poses  # noqa: E402

# validated two-series palette (dataviz reference slots 1-2) and chart chrome
NGP, GS = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHODS = [("instant-ngp", "InstantNGP", NGP), ("3dgs", "3DGS", GS)]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", [GS, "#f0efec", NGP])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def image_axis(ax, img, title=None, **kw):
    ax.imshow(img, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, fontweight="normal", color=INK2)


def abs_err(a, b):
    return np.abs(a - b).mean(2)


class Data:
    def __init__(self, a):
        self.a = a
        self.summary = json.loads((Path(a.results) / "summary.json").read_text())
        self.names = [Path(l.strip()).stem for l in Path(a.test_list).read_text().splitlines() if l.strip()]
        self.gt_dir = Path(a.scene) / "images"
        gs_it = self.summary["3dgs"].get("final", {}).get("iteration")
        ngp_it = self.summary["instant-ngp"].get("final", {}).get("iteration")
        self.pred_dirs = {
            "3dgs": Path(a.outputs) / "eval" / "gs" / f"iter_{gs_it:06d}" if gs_it else None,
            "instant-ngp": Path(a.outputs) / "eval" / "ngp" / f"step_{ngp_it:06d}" if ngp_it else None,
        }
        self._cache = {}

    def gt(self, n):
        return self._get(("gt", n), lambda: load(next(self.gt_dir.glob(f"{n}.*"))))

    def pred(self, m, n):
        if self.pred_dirs[m] is None:
            raise FileNotFoundError(f"no final renders for {m}")
        return self._get((m, n), lambda: load(self.pred_dirs[m] / f"{n}.png"))

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def per_view(self, m, metric):
        pv = self.summary[m].get("per_view", {})
        return [pv.get(n, {}).get(metric) for n in self.names]


# ---------------------------------------------------------------- test-view images

def error_map_views(d):
    """The view with the largest PSNR gap between the methods and the worst view (mean PSNR)."""
    pv = {m: d.summary[m]["per_view"] for m, _, _ in METHODS}
    gap = {n: pv["3dgs"][n]["psnr"] - pv["instant-ngp"][n]["psnr"] for n in d.names}
    mean = {n: np.mean([pv[m][n]["psnr"] for m in pv]) for n in d.names}
    picks = [max(gap, key=gap.get), min(mean, key=mean.get)]
    return [n for n in d.names if n in picks]


def fig_test_grid(d, out):
    vmax = 0.25
    names = error_map_views(d)
    fig, axes = plt.subplots(len(names), 5, figsize=(10.5, 3.9 * len(names)), squeeze=False, layout="constrained")
    for r, n in enumerate(names):
        gt = d.gt(n)
        image_axis(axes[r, 0], gt, f"Ground truth\nview {n}")
        for c, (m, label, _) in enumerate(METHODS):
            img = d.pred(m, n)
            pv = d.summary[m]["per_view"][n]
            image_axis(axes[r, 1 + c], img, f"{label}\n{pv['psnr']:.2f} dB · SSIM {pv['ssim']:.3f}")
            image_axis(axes[r, 3 + c], abs_err(img, gt), f"|error| {label}", cmap=SEQ, vmin=0, vmax=vmax)
            im = axes[r, 3 + c].images[-1]
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.01, shrink=0.5)
    cb.set_label("mean absolute RGB error (0-1)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Test views: ground truth, renders and error maps (same error scale everywhere)", x=0.01, ha="left", fontweight="bold")
    save(fig, out, "01_test_views")


def detail_crops(gt, size, k=2):
    g = gaussian_filter(np.abs(laplace(gt.mean(2))), 6)
    h, w = g.shape
    boxes = []
    for _ in range(k):
        g2 = g.copy()
        g2[: size // 2], g2[h - size // 2:], g2[:, : size // 2], g2[:, w - size // 2:] = 0, 0, 0, 0
        for (y0, x0) in boxes:
            g2[max(0, y0 - size // 2): y0 + size + size // 2, max(0, x0 - size // 2): x0 + size + size // 2] = 0
        cy, cx = np.unravel_index(np.argmax(g2), g2.shape)
        boxes.append((int(cy - size // 2), int(cx - size // 2)))
    return boxes


def fig_zoom(d, out):
    gap = {n: abs((d.summary["instant-ngp"]["per_view"][n]["psnr"]) - d.summary["3dgs"]["per_view"][n]["psnr"]) for n in d.names}
    mean_psnr = {n: np.mean([d.summary[m]["per_view"][n]["psnr"] for m, _, _ in METHODS]) for n in d.names}
    picks = []
    for n in sorted(gap, key=gap.get, reverse=True)[:2] + [min(mean_psnr, key=mean_psnr.get)]:
        if n not in picks:
            picks.append(n)
    size = 150
    fig, axes = plt.subplots(len(picks), 7, figsize=(14, 3.0 * len(picks)), squeeze=False, layout="constrained",
                             gridspec_kw={"width_ratios": [0.62, 1, 1, 1, 1, 1, 1]})
    for r, n in enumerate(picks):
        gt = d.gt(n)
        boxes = detail_crops(gt, size)
        image_axis(axes[r, 0], gt, f"view {n}")
        for b, (y, x) in enumerate(boxes):
            axes[r, 0].add_patch(Rectangle((x, y), size, size, fill=False, ec=INK, lw=1.2))
            axes[r, 0].text(x + 4, y - 8, "AB"[b], color=INK, fontsize=8, fontweight="bold")
            for c, (label, img) in enumerate([("GT", gt), ("InstantNGP", d.pred("instant-ngp", n)), ("3DGS", d.pred("3dgs", n))]):
                crop = img[y:y + size, x:x + size]
                image_axis(axes[r, 1 + 3 * b + c], crop, f"{'AB'[b]}: {label}", interpolation="nearest")
    fig.suptitle("Zoomed detail: views with the largest quality gap between the methods, and the hardest view",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, out, "02_zoom_crops")


# ---------------------------------------------------------------- metric plots

def series_label(ax, x, y, text, color):
    if y is not None:
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


def fig_per_view(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(d.names))
    for ax, (metric, title) in zip(axes, [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips_vgg", "LPIPS-VGG ↓")]):
        for m, label, color in METHODS:
            y = d.per_view(m, metric)
            if any(v is None for v in y):
                continue
            ax.plot(x, y, color=color, marker="o", ms=5, lw=1.6, label=label)
            mean = d.summary[m]["final"][metric]
            ax.axhline(mean, color=color, lw=0.8, alpha=0.5)
        ax.set_xticks(x, d.names, rotation=45)
        ax.set_xlabel("test view (image name)")
        ax.set_title(title, loc="left")
    axes[0].legend(loc="lower left")
    fig.suptitle("Per-view quality on the 8 held-out views (thin lines = mean)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "04_per_view_metrics")



def fig_convergence(d, out):
    metrics = [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips_vgg", "LPIPS-VGG ↓")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for col, (metric, title) in enumerate(metrics):
        for m, label, color in METHODS:
            conv = [c for c in d.summary[m].get("convergence", []) if c.get(metric) is not None]
            if not conv:
                continue
            it = [c["iteration"] for c in conv]
            y = [c[metric] for c in conv]
            axes[0, col].plot(it, y, color=color, marker="o", label=label)
            series_label(axes[0, col], it[-1], y[-1], f"{y[-1]:.3f}" if metric != "psnr" else f"{y[-1]:.2f}", color)
            t = [c.get("wall_s") for c in conv]
            if all(v is not None for v in t):
                axes[1, col].plot(np.array(t) / 60, y, color=color, marker="o", label=label)
        axes[0, col].set_xscale("log")
        axes[0, col].set_xlabel("training iterations (log scale)")
        axes[0, col].set_title(title, loc="left")
        axes[1, col].set_xlabel("training wall-clock time (min)")
        axes[1, col].set_title(f"{title} vs time", loc="left")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Convergence: test-set quality at each saved checkpoint", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "05_convergence")


def bar_panel(ax, values, title, unit, fmt_s, log=False):
    labels = [label for _, label, _ in METHODS]
    colors = [c for _, _, c in METHODS]
    vals = [values.get(m) for m, _, _ in METHODS]
    shown = [v if v is not None else 0 for v in vals]
    bars = ax.bar(labels, shown, color=colors, width=0.55)
    if log:
        ax.set_yscale("log")
    for b, v in zip(bars, vals):
        txt = "n/a" if v is None else fmt_s.format(v)
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height() if v else 0), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_title(title, loc="left")
    ax.set_ylabel(unit)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)


def fig_efficiency(d, out):
    s = d.summary
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    bar_panel(axes[0], {m: (s[m].get("train_wall_s") or 0) / 60 or None for m, _, _ in METHODS},
              "Training time", "minutes (wall clock)", "{:.1f} min")
    bar_panel(axes[1], {m: s[m].get("render", {}).get("fps") for m, _, _ in METHODS},
              "Rendering speed", "frames / second (log)", "{:.1f} FPS", log=True)
    bar_panel(axes[2], {m: (s[m].get("model_bytes") or 0) / 2 ** 20 or None for m, _, _ in METHODS},
              "Model size on disk", "MB (log)", "{:.1f} MB", log=True)
    per_process = all(s[m].get("peak_proc_mib") for m, _, _ in METHODS)
    key = "peak_proc_mib" if per_process else "peak_total_mib"
    bar_panel(axes[3], {m: s[m].get(key) for m, _, _ in METHODS},
              "Peak GPU memory (training)" if per_process else "Peak GPU memory, whole GPU", "MiB", "{:.0f} MiB")
    ngp_r = s["instant-ngp"].get("render", {})
    res = f"{ngp_r.get('width', '?')}×{ngp_r.get('height', '?')}"
    fig.suptitle(f"Efficiency (render speed at {res}, InstantNGP at {ngp_r.get('spp', 1)} spp; "
                 f"GPU: {ngp_r.get('gpu') or s['3dgs'].get('render', {}).get('gpu', 'n/a')})",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "06_efficiency")


def fig_model_growth(d, out):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for m, label, color in METHODS:
        conv = [c for c in d.summary[m].get("convergence", []) if c.get("bytes")]
        if conv:
            x, y = [c["iteration"] for c in conv], [c["bytes"] / 2 ** 20 for c in conv]
            ax.plot(x, y, color=color, marker="o", label=label)
            series_label(ax, x[-1], y[-1], f"{y[-1]:.1f} MB", color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training iterations (log)")
    ax.set_ylabel("model file size, MB (log)")
    ax.legend(loc="upper left")
    ax.set_title("Model size during training: 3DGS grows by densification, InstantNGP is fixed", loc="left")
    save(fig, out, "07_model_growth")


def fig_error_distribution(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = np.linspace(0, 0.5, 101)
    for m, label, color in METHODS:
        e = np.concatenate([abs_err(d.pred(m, n), d.gt(n)).ravel() for n in d.names])
        h, _ = np.histogram(e, bins=bins, density=True)
        axes[0].plot(bins[:-1] + np.diff(bins) / 2, h, color=color, lw=1.6, label=label)
        xs = np.sort(e)[:: max(1, e.size // 4000)]
        p95 = float(np.percentile(e, 95))
        axes[1].plot(xs, np.linspace(0, 1, xs.size), color=color, lw=1.6,
                     label=f"{label}: median {np.median(e):.3f}, 95th pct {p95:.3f}")
    axes[1].legend(loc="lower right")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("per-pixel absolute error")
    axes[0].set_ylabel("density (log)")
    axes[0].set_title("Error histogram over all test pixels", loc="left")
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("per-pixel absolute error (log)")
    axes[1].set_ylabel("fraction of pixels ≤ error")
    axes[1].set_title("Cumulative error distribution", loc="left")
    fig.tight_layout()
    save(fig, out, "08_error_distribution")


def fig_extrapolation(d, out):
    _, colmap_c2w, _, _ = poses.read_colmap_text(Path(d.a.scene) / "colmap_text")
    train = [Path(l.strip()).stem for l in Path(d.a.train_list).read_text().splitlines() if l.strip()]
    centre = {Path(k).stem: v[:3, 3] for k, v in colmap_c2w.items()}
    tc = np.array([centre[n] for n in train])
    spacing = np.median([np.linalg.norm(tc - c, axis=1)[np.argsort(np.linalg.norm(tc - c, axis=1))[1]] for c in tc])
    dist = [np.linalg.norm(tc - centre[n], axis=1).min() / spacing for n in d.names]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m, label, color in METHODS:
        y = d.per_view(m, "psnr")
        ax.scatter(dist, y, color=color, s=40, label=label, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for n, x in zip(d.names, dist):
        ax.annotate(n, (x, max(d.per_view("instant-ngp", "psnr")[d.names.index(n)], d.per_view("3dgs", "psnr")[d.names.index(n)])),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    ax.set_xlabel("distance to nearest training camera (in median training-camera spacings)")
    ax.set_ylabel("PSNR (dB)")
    ax.legend()
    ax.set_title("Does quality drop for test views farther from the training cameras?", loc="left")
    save(fig, out, "09_view_extrapolation")


def sharpness(img):
    return float(laplace(img.mean(2)).var())


def fig_sharpness(d, out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(d.names))
    w = 0.36
    for k, (m, label, color) in enumerate(METHODS):
        r = [sharpness(d.pred(m, n)) / sharpness(d.gt(n)) for n in d.names]
        ax.bar(x + (k - 0.5) * w, r, width=w - 0.04, color=color, label=f"{label} (mean {np.mean(r):.2f})")
    ax.axhline(1.0, color=INK2, lw=1)
    ax.annotate("as sharp as ground truth", (x[-1] + 0.5, 1.0), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(x, d.names)
    ax.set_xlabel("test view")
    ax.set_ylabel("Laplacian variance, render / GT")
    ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Sharpness of the renders relative to the ground truth (below 1 = blurrier)", loc="left")
    save(fig, out, "10_sharpness")


def radial_spectrum(img, nbins=60):
    g = img.mean(2)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(g.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(g.shape[1]))
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    s = np.bincount(idx[ok], p.ravel()[ok], nbins) / np.maximum(np.bincount(idx[ok], minlength=nbins), 1)
    return edges[:-1] + np.diff(edges) / 2, s


def fig_spectrum(d, out):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m, label, color in METHODS:
        ratios = []
        for n in d.names:
            f, sp = radial_spectrum(d.pred(m, n))
            _, sg = radial_spectrum(d.gt(n))
            ratios.append(sp / np.maximum(sg, 1e-12))
        r = np.exp(np.log(np.array(ratios)).mean(0))
        ax.plot(f[1:], r[1:], color=color, label=label)
        series_label(ax, f[-1], r[-1], f"{r[-1]:.2f}", color)
    ax.axhline(1.0, color=INK2, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("spatial frequency (cycles / pixel)")
    ax.set_ylabel("power: render / GT (log, geometric mean)")
    ax.legend(loc="lower left")
    ax.set_title("Frequency content: values below 1 mean lost fine detail", loc="left")
    save(fig, out, "11_frequency_spectrum")


# ---------------------------------------------------------------- novel views

def novel_dirs(d):
    root = Path(d.a.outputs) / "novel"
    paths = json.loads((Path(d.a.scene) / "novel" / "paths_summary.json").read_text())
    avail = {}
    for name in paths:
        dirs = {m: root / ("ngp" if m == "instant-ngp" else "gs") / name for m, _, _ in METHODS}
        frames = [sorted(p.glob("*.png")) for p in dirs.values()]
        if all(frames) and len(frames[0]) == len(frames[1]):
            avail[name] = dict(zip(dirs, frames))
        else:
            print(f"  (novel path {name}: renders missing or frame counts differ, skipped)")
    return avail, paths


def extreme_index(name, n):
    """Frame index showing the largest deviation from the captured views for each path type."""
    return {"dolly": n // 4, "orbit": n // 4, "look_around": n // 4}.get(name, n // 2)


def fig_novel(d, out):
    avail, info = novel_dirs(d)
    if not avail:
        raise FileNotFoundError("no novel-view renders")
    names = list(avail)
    fig, axes = plt.subplots(2, len(names), figsize=(2.1 * len(names), 7.6), squeeze=False, layout="constrained")
    for c, name in enumerate(names):
        k = extreme_index(name, len(avail[name]["3dgs"]))
        for r, (m, label, _) in enumerate(METHODS):
            image_axis(axes[r, c], load(avail[name][m][k]), f"{name}\n{label}" if r == 0 else label)
    fig.suptitle("Novel views far from the capture path (one frame per path, same camera for both methods)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, out, "12_novel_overview")

    for name, frames in avail.items():
        n = len(frames["3dgs"])
        idx = np.linspace(0, n - 1, 6).round().astype(int)
        fig, axes = plt.subplots(2, len(idx), figsize=(2.1 * len(idx), 7.6), squeeze=False, layout="constrained")
        for c, k in enumerate(idx):
            for r, (m, label, _) in enumerate(METHODS):
                image_axis(axes[r, c], load(frames[m][k]), f"{label} · frame {k}")
        fig.suptitle(f"Novel path '{name}': {info[name]['description']}", x=0.01, ha="left", fontweight="bold")
        save(fig, out, f"13_novel_{name}")


def fig_camera_paths(d, out):
    _, colmap_c2w, pts, cols = poses.read_colmap_text(Path(d.a.scene) / "colmap_text")
    names = sorted(colmap_c2w)
    c2ws = np.array([colmap_c2w[n] for n in names])
    fr = poses.scene_frame(c2ws, pts)

    def coords(x):
        c = x - fr["centre"]
        return c @ fr["ex"], -(c @ fr["ey"]), c @ fr["ez"]

    test = {Path(l.strip()).name for l in Path(d.a.test_list).read_text().splitlines() if l.strip()}
    novel = {}
    for p in sorted((Path(d.a.scene) / "novel").glob("*.json")):
        doc = json.loads(p.read_text())
        if "path" in doc:
            novel[doc["path"]] = np.array([f["colmap_c2w"] for f in doc["frames"]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    px, py, pz = coords(pts)
    keep = (np.abs(pz) < 0.5 * fr["depth"]) & (np.abs(py) < 0.5 * fr["depth"])
    path_colors = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", MUTED]
    for ax, (a, b, ylab, title) in zip(axes, [(pz, None, "depth into the scene", "Top view"), (py, None, "height", "Front view")]):
        ax.scatter(px[keep], a[keep], s=0.4, c=cols[keep], alpha=0.6, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlabel("along the capture path")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    for i, n in enumerate(names):
        x, y, z = coords(c2ws[i:i + 1, :3, 3])
        is_test = n in test
        kw = dict(s=34 if is_test else 18, marker="s" if is_test else "^", c=GS if is_test else INK, zorder=4,
                  edgecolors=SURFACE, linewidths=1)
        axes[0].scatter(x, z, **kw)
        axes[1].scatter(x, y, **kw)
    for (name, mats), color in zip(novel.items(), path_colors):
        x, y, z = coords(mats[:, :3, 3])
        if np.ptp(x) + np.ptp(y) + np.ptp(z) < 1e-6 * fr["depth"]:   # rotation-only path: draw its position
            axes[0].scatter(x[:1], z[:1], marker="*", s=140, color=color, label=f"{name} (rotates in place)", zorder=5)
            axes[1].scatter(x[:1], y[:1], marker="*", s=140, color=color, zorder=5)
            continue
        axes[0].plot(x, z, color=color, lw=1.4, label=name)
        axes[1].plot(x, y, color=color, lw=1.4)
    axes[0].scatter([], [], marker="^", c=INK, label="training cameras")
    axes[0].scatter([], [], marker="s", c=GS, label="test cameras")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=5, fontsize=8)
    fig.suptitle("Camera layout: captured training/test cameras and the novel-view paths (COLMAP points coloured)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, out, "14_camera_paths")


def read_gs_ply(path):
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError("expected a binary little-endian PLY")
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    return data


def fig_gaussian_stats(d, out):
    gs = d.summary["3dgs"]
    it = gs.get("final", {}).get("iteration")
    ply = Path(d.a.outputs) / "gs" / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    g = read_gs_ply(ply)
    opacity = 1 / (1 + np.exp(-g["opacity"]))
    scale = np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(opacity, bins=50, color=GS)
    axes[0].set_xlabel("opacity")
    axes[0].set_ylabel("Gaussians")
    axes[0].set_title("Opacity distribution", loc="left")
    axes[1].hist(np.log10(scale.max(1)), bins=60, color=GS)
    axes[1].set_xlabel("log10 largest axis length (COLMAP units)")
    axes[1].set_title("Gaussian size distribution", loc="left")
    aniso = scale.max(1) / np.maximum(scale.min(1), 1e-12)
    axes[2].hist(np.log10(aniso), bins=60, color=GS)
    axes[2].set_xlabel("log10 (largest / smallest axis)")
    axes[2].set_title("Anisotropy (flat, needle-like splats)", loc="left")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"3DGS model at iteration {it}: {len(g):,} Gaussians, median opacity {np.median(opacity):.2f}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "15_gaussian_stats")


# ---------------------------------------------------------------- videos

def make_videos(d, vid_dir, fps):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("  ffmpeg not found: videos skipped")
        return
    avail, _ = novel_dirs(d)
    vid_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for name, frames in avail.items():
        with tempfile.TemporaryDirectory() as tmp:
            for k, (fa, fb) in enumerate(zip(frames["instant-ngp"], frames["3dgs"])):
                a, b = Image.open(fa).convert("RGB"), Image.open(fb).convert("RGB")
                bar = 54
                canvas = Image.new("RGB", (a.width + b.width + 12, a.height + bar), SURFACE)
                canvas.paste(a, (0, bar))
                canvas.paste(b, (a.width + 12, bar))
                draw = ImageDraw.Draw(canvas)
                draw.text((14, 10), "InstantNGP", fill=NGP, font=font)
                draw.text((a.width + 26, 10), "3DGS", fill=GS, font=font)
                canvas.save(Path(tmp) / f"{k:05d}.png")
            target = vid_dir / f"{name}_side_by_side.mp4"
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(tmp) / "%05d.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(target)]
            subprocess.run(cmd, check=True)
            print(f"  {target.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--results", default="results")
    p.add_argument("--scene", default="data/scene")
    p.add_argument("--test_list", default="results/test_list.txt")
    p.add_argument("--train_list", default="results/train_list.txt")
    p.add_argument("--fig_dir", default="results/figures")
    p.add_argument("--video_dir", default="results/videos")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no_video", action="store_true")
    a = p.parse_args()

    d = Data(a)
    out = Path(a.fig_dir)
    jobs = [fig_test_grid, fig_zoom, fig_per_view, fig_convergence, fig_efficiency,
            fig_model_growth, fig_error_distribution, fig_extrapolation, fig_sharpness, fig_spectrum,
            fig_novel, fig_camera_paths, fig_gaussian_stats]
    skipped = []
    for job in jobs:
        try:
            job(d, out)
        except (FileNotFoundError, KeyError, StopIteration) as e:
            plt.close("all")
            skipped.append(f"{job.__name__}: {type(e).__name__} {e}")
    if not a.no_video:
        try:
            make_videos(d, Path(a.video_dir), a.fps)
        except FileNotFoundError as e:
            skipped.append(f"videos: {e}")
    if skipped:
        print("skipped (missing inputs):")
        for s in skipped:
            print("  -", s)


if __name__ == "__main__":
    main()
