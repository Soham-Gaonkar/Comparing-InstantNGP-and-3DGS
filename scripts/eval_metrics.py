"""PSNR / SSIM / LPIPS of rendered test views against the ground-truth images.

One script for both methods so the numbers are directly comparable.
  PSNR : 10 log10(1 / MSE) per image, images in [0, 1]
  SSIM : identical to 3DGS utils/loss_utils.ssim (11x11 Gaussian window, sigma 1.5,
         zero padding, mean over the map)
  LPIPS: official `lpips` package (v0.1), VGG and AlexNet backbones, inputs mapped
         to [-1, 1] as the package expects.

Renders are matched to ground truth by file stem: <pred_dir>/0000.png <-> <gt_dir>/0000.jpg
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg")


def load_rgb(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()


def find_image(folder, stem):
    for ext in IMG_EXT:
        p = Path(folder) / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image named {stem}.* in {folder}")


def gaussian_window(size=11, sigma=1.5, channels=3):
    g = torch.tensor([math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
    g = (g / g.sum()).unsqueeze(1)
    w = (g @ g.t()).float()
    return w.expand(channels, 1, size, size).contiguous()


def ssim(img1, img2, window):
    pad = window.shape[-1] // 2
    ch = img1.shape[1]
    mu1 = F.conv2d(img1, window, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, window, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=ch) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


def psnr(img1, img2):
    mse = float(((img1 - img2) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", default="data/scene/images")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--list", default="results/test_list.txt", help="test image names, one per line")
    p.add_argument("--out", required=True, help="output json")
    p.add_argument("--method", default="")
    p.add_argument("--tag", default="", help="free-form label, e.g. the iteration")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_lpips", action="store_true")
    args = p.parse_args()

    names = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    dev = torch.device(args.device)
    window = gaussian_window().to(dev)

    nets = {}
    if not args.no_lpips:
        import lpips
        nets = {k: lpips.LPIPS(net=k, verbose=False).to(dev).eval() for k in ("vgg", "alex")}

    per_view = {}
    with torch.no_grad():
        for name in names:
            stem = Path(name).stem
            gt = load_rgb(find_image(args.gt_dir, stem)).to(dev)
            pred = load_rgb(find_image(args.pred_dir, stem)).to(dev)
            if gt.shape != pred.shape:
                raise ValueError(f"{stem}: render {tuple(pred.shape[2:])} vs ground truth {tuple(gt.shape[2:])}")
            row = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt, window)}
            for k, net in nets.items():
                row[f"lpips_{k}"] = float(net(pred * 2 - 1, gt * 2 - 1).mean())
            per_view[stem] = row
            print(stem, "  ".join(f"{k}={v:.4f}" for k, v in row.items()))

    keys = list(next(iter(per_view.values())).keys())
    mean = {k: float(np.mean([r[k] for r in per_view.values()])) for k in keys}
    # PSNR of the mean MSE, the other common convention
    mean_mse = float(np.mean([10 ** (-r["psnr"] / 10) for r in per_view.values()]))
    mean["psnr_of_mean_mse"] = float("inf") if mean_mse == 0 else -10 * math.log10(mean_mse)
    out = {"method": args.method, "tag": args.tag, "pred_dir": str(args.pred_dir), "n_views": len(per_view),
           "mean": mean, "per_view": per_view}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("mean", "  ".join(f"{k}={v:.4f}" for k, v in mean.items()), "->", args.out)


if __name__ == "__main__":
    main()
