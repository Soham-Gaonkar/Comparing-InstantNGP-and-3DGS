"""Gather metrics, timings, model sizes and GPU memory of both methods into one summary.

Reads (anything missing is reported and skipped):
  results/metrics/gs_iter_XXXXXX.json, results/metrics/ngp_step_XXXXXX.json   (eval_metrics.py)
  results/timing/gs.json, results/timing/ngp.json                             (render_*.py --bench)
  logs/{gs,ngp}_train_time.json, logs/{gs,ngp}_gpumem.csv                     (run_*.sh)
  outputs/gs/point_cloud/iteration_*/point_cloud.ply                           (3DGS train.py)
  outputs/gs/results.json                                                      (official 3DGS metrics.py, optional)
  outputs/ngp/train_log.json                                                   (train_ngp.py)
Writes results/summary.json and results/summary.md
"""
import argparse
import json
import re
from pathlib import Path

METRICS = ["psnr", "ssim", "lpips_vgg", "lpips_alex"]


def read_json(path):
    path = Path(path)
    if not path.exists():
        print(f"  missing: {path}")
        return None
    return json.loads(path.read_text())


def ply_vertex_count(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no PLY header end")
            header += line
    m = re.search(rb"element vertex (\d+)", header)
    return int(m.group(1)) if m else None


def gpu_peaks(path):
    path = Path(path)
    if not path.exists():
        print(f"  missing: {path}")
        return {}
    peaks = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) == 3 and parts[2].strip().isdigit():
            peaks[parts[1]] = max(peaks.get(parts[1], 0), int(parts[2]))
    return {f"peak_{k}_mib": v for k, v in peaks.items()}


def metric_series(metric_dir, prefix):
    rows = []
    for p in sorted(Path(metric_dir).glob(f"{prefix}_*.json")):
        m = re.search(r"_(\d+)\.json$", p.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        rows.append({"iteration": int(m.group(1)), **{k: d["mean"].get(k) for k in METRICS}, "per_view": d["per_view"]})
    return sorted(rows, key=lambda r: r["iteration"])


def collect_gs(a):
    out = {"name": "3DGS"}
    t = read_json(Path(a.logs) / "gs_train_time.json")
    conv = metric_series(Path(a.results) / "metrics", "gs_iter")
    ckpt = {}
    for ply in Path(a.outputs, "gs", "point_cloud").glob("iteration_*/point_cloud.ply"):
        it = int(ply.parent.name.split("_")[1])
        ckpt[it] = {"bytes": ply.stat().st_size, "n_gaussians": ply_vertex_count(ply),
                    "wall_s": (ply.stat().st_mtime - t["start"]) if t else None}
    for r in conv:
        r.update({k: ckpt.get(r["iteration"], {}).get(k) for k in ("bytes", "n_gaussians", "wall_s")})
    out["convergence"] = conv
    if conv:
        final = conv[-1]
        out["final"] = {k: final[k] for k in METRICS} | {"iteration": final["iteration"]}
        out["per_view"] = final["per_view"]
        out["model_bytes"] = final.get("bytes")
        out["n_gaussians"] = final.get("n_gaussians")
    if t:
        out["train_wall_s"] = t["wall_s"]
    timing = read_json(Path(a.results) / "timing" / "gs.json")
    if timing:
        out["render"] = {"ms": timing["ms_mean"], "ms_std": timing["ms_std"], "fps": timing["fps_mean"],
                         "width": timing["width"], "height": timing["height"], "gpu": timing["gpu"]}
    out.update(gpu_peaks(Path(a.logs) / "gs_gpumem.csv"))
    official = read_json(Path(a.outputs) / "gs" / "results.json")
    if official:
        out["official_3dgs_metrics"] = next(iter(official.values()))
    return out


def collect_ngp(a):
    out = {"name": "InstantNGP"}
    t = read_json(Path(a.logs) / "ngp_train_time.json")
    log = read_json(Path(a.outputs) / "ngp" / "train_log.json")
    conv = metric_series(Path(a.results) / "metrics", "ngp_step")
    ckpt = {c["target_step"]: c for c in (log or {}).get("checkpoints", [])}
    for r in conv:
        c = ckpt.get(r["iteration"], {})
        r.update({"bytes": c.get("snapshot_bytes"), "wall_s": c.get("wall_s"), "train_s": c.get("train_s")})
    out["convergence"] = conv
    if conv:
        final = conv[-1]
        out["final"] = {k: final[k] for k in METRICS} | {"iteration": final["iteration"]}
        out["per_view"] = final["per_view"]
        out["model_bytes"] = final.get("bytes")
    if log:
        out["n_params"] = log.get("n_params")
        out["n_encoding_params"] = log.get("n_encoding_params")
        out["model_bytes_uncompressed"] = log.get("final_uncompressed_bytes")
        out["train_loop_s"] = log.get("train_s")
    if t:
        out["train_wall_s"] = t["wall_s"]
    timing = read_json(Path(a.results) / "timing" / "ngp.json")
    if timing:
        spp = timing["spp"]
        out["render_by_spp"] = spp
        key = "1" if "1" in spp else sorted(spp, key=int)[0]
        out["render"] = {"ms": spp[key]["ms_mean"], "ms_std": spp[key]["ms_std"], "fps": spp[key]["fps_mean"],
                         "spp": int(key), "width": timing["width"], "height": timing["height"], "gpu": timing["gpu"]}
    out.update(gpu_peaks(Path(a.logs) / "ngp_gpumem.csv"))
    return out


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def markdown(summary):
    gs, ngp = summary["3dgs"], summary["instant-ngp"]
    rows = [("PSNR (dB) ↑", "psnr", ".2f"), ("SSIM ↑", "ssim", ".4f"),
            ("LPIPS-VGG ↓", "lpips_vgg", ".4f"), ("LPIPS-Alex ↓", "lpips_alex", ".4f")]
    lines = ["# Results summary", "", "## Test-view quality (mean over test views)", "",
             "| Metric | InstantNGP | 3DGS |", "|---|---|---|"]
    for label, k, spec in rows:
        lines.append(f"| {label} | {fmt(ngp.get('final', {}).get(k), spec)} | {fmt(gs.get('final', {}).get(k), spec)} |")
    mb = lambda b: None if b is None else b / 2 ** 20
    lines += ["", "## Efficiency", "", "| | InstantNGP | 3DGS |", "|---|---|---|",
              f"| Training iterations | {fmt(ngp.get('final', {}).get('iteration'), 'd')} | {fmt(gs.get('final', {}).get('iteration'), 'd')} |",
              f"| Training time, wall clock (s) | {fmt(ngp.get('train_wall_s'), '.1f')} | {fmt(gs.get('train_wall_s'), '.1f')} |",
              f"| Render time (ms / frame) | {fmt(ngp.get('render', {}).get('ms'), '.2f')} (spp {ngp.get('render', {}).get('spp', '–')}) | {fmt(gs.get('render', {}).get('ms'), '.2f')} |",
              f"| Render speed (FPS) | {fmt(ngp.get('render', {}).get('fps'), '.1f')} | {fmt(gs.get('render', {}).get('fps'), '.1f')} |",
              f"| Model size on disk (MB) | {fmt(mb(ngp.get('model_bytes')), '.2f')} (.ingp, compressed) | {fmt(mb(gs.get('model_bytes')), '.2f')} (.ply) |",
              f"| Model size uncompressed (MB) | {fmt(mb(ngp.get('model_bytes_uncompressed')), '.2f')} | {fmt(mb(gs.get('model_bytes')), '.2f')} |",
              f"| Parameters | {fmt(ngp.get('n_params'), ',d')} | {fmt(None if gs.get('n_gaussians') is None else gs['n_gaussians'] * 59, ',d')} ({fmt(gs.get('n_gaussians'), ',d')} Gaussians × 59) |",
              f"| Peak GPU memory, process (MiB) | {fmt(ngp.get('peak_proc_mib'), 'd')} | {fmt(gs.get('peak_proc_mib'), 'd')} |",
              f"| Peak GPU memory, whole GPU (MiB) | {fmt(ngp.get('peak_total_mib'), 'd')} | {fmt(gs.get('peak_total_mib'), 'd')} |"]
    gpu = gs.get("render", {}).get("gpu") or ngp.get("render", {}).get("gpu")
    if gpu:
        lines += ["", f"GPU: {gpu}"]
    off = gs.get("official_3dgs_metrics")
    if off:
        lines += ["", "Cross-check with the official 3DGS metrics.py on the same renders: "
                  f"PSNR {off.get('PSNR', float('nan')):.3f}, SSIM {off.get('SSIM', float('nan')):.4f} "
                  f"(ours {fmt(gs.get('final', {}).get('psnr'), '.3f')}, {fmt(gs.get('final', {}).get('ssim'), '.4f')}). "
                  "Its LPIPS differs because metrics.py feeds [0,1] images to a network expecting [-1,1]."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--results", default="results")
    p.add_argument("--logs", default="logs")
    a = p.parse_args()
    summary = {"3dgs": collect_gs(a), "instant-ngp": collect_ngp(a)}
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "summary.json").write_text(json.dumps(summary, indent=2))
    md = markdown(summary)
    (Path(a.results) / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
