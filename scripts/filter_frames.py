import argparse
import csv
import shutil
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def sharpness(path):
    # variance of Laplacian: higher = sharper
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/candidates")
    p.add_argument("--dst", default="data/scene/input")
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--results", default="results")
    args = p.parse_args()

    src, dst, res = Path(args.src), Path(args.dst), Path(args.results)
    dst.mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(parents=True, exist_ok=True)

    frames = sorted(src.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no .jpg files in {src}")

    scores = [sharpness(f) for f in frames]
    print(f"scored {len(frames)} frames")

    # keep the sharpest frame in each consecutive group
    kept = []
    for start in range(0, len(frames), args.group):
        idx = range(start, min(start + args.group, len(frames)))
        kept.append(max(idx, key=lambda i: scores[i]))

    # copy with sequential names so sorted order = capture order
    for new_i, i in enumerate(kept):
        shutil.copy2(frames[i], dst / f"{new_i:04d}.jpg")
    print(f"kept {len(kept)} frames -> {dst}")

    # save scores for the report
    kept_set = set(kept)
    with open(res / "sharpness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "score", "kept"])
        for i, fr in enumerate(frames):
            w.writerow([fr.name, f"{scores[i]:.2f}", int(i in kept_set)])

    # plot all scores and kept frames highlighted
    plt.figure(figsize=(10, 4))
    plt.plot(scores, color="lightgray", label="all candidates")
    plt.scatter(kept, [scores[i] for i in kept], s=12, color="tab:blue", label="kept")
    plt.xlabel("candidate frame index")
    plt.ylabel("variance of Laplacian")
    plt.title(f"Frame sharpness (group size {args.group})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(res / "figures" / "sharpness.png", dpi=150)
    print(f"plot -> {res / 'figures' / 'sharpness.png'}")


if __name__ == "__main__":
    main()
