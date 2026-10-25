"""Side-by-side GIFs of the novel-view renders for the README (left InstantNGP, right 3DGS).

Reads the frames written by render_ngp.py / render_gs.py (outputs/novel/{ngp,gs}/<path>/0000.png ...)
and writes results/media/<path>.gif with ffmpeg (two-pass palette for clean colours).

  python scripts/make_gifs.py --novel outputs/novel --out results/media
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NGP, GS = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"


def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--novel", default="outputs/novel")
    p.add_argument("--out", default="results/media")
    p.add_argument("--paths", nargs="+", default=["trajectory", "orbit", "shift_up"])
    p.add_argument("--pane_width", type=int, default=440, help="width of each render in the GIF (source is 701)")
    p.add_argument("--step", type=int, default=2, help="use every n-th rendered frame")
    p.add_argument("--delay_ms", type=int, default=140, help="display time per GIF frame")
    args = p.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    font = load_font(26)
    gap, bar = 10, 42

    for name in args.paths:
        fa = sorted((Path(args.novel) / "ngp" / name).glob("*.png"))[::args.step]
        fb = sorted((Path(args.novel) / "gs" / name).glob("*.png"))[::args.step]
        if not fa or len(fa) != len(fb):
            raise SystemExit(f"{name}: {len(fa)} InstantNGP vs {len(fb)} 3DGS frames")
        with tempfile.TemporaryDirectory() as tmp:
            for k, (a, b) in enumerate(zip(fa, fb)):
                a, b = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
                h = round(a.height * args.pane_width / a.width)
                a = a.resize((args.pane_width, h), Image.LANCZOS)
                b = b.resize((args.pane_width, h), Image.LANCZOS)
                canvas = Image.new("RGB", (2 * args.pane_width + gap, h + bar), SURFACE)
                canvas.paste(a, (0, bar))
                canvas.paste(b, (args.pane_width + gap, bar))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 7), "InstantNGP", fill=NGP, font=font)
                draw.text((args.pane_width + gap + 12, 7), "3DGS", fill=GS, font=font)
                canvas.save(Path(tmp) / f"{k:05d}.png")
            target = out / f"{name}.gif"
            fps = 1000.0 / args.delay_ms
            graph = ("split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=full[p];"
                     "[s1][p]paletteuse=dither=bayer:bayer_scale=5")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:.4f}",
                            "-i", str(Path(tmp) / "%05d.png"), "-filter_complex", graph,
                            "-loop", "0", str(target)], check=True)
            print(f"{target}: {len(fa)} frames, {len(fa) * args.delay_ms / 1000:.1f} s loop, "
                  f"{canvas.width}x{canvas.height}, {target.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
