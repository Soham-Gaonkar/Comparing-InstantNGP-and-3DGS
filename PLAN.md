# Assignment 2 – InstantNGP vs 3D Gaussian Splatting: Step-by-Step Plan

> Due: **Oct 25, 2026**. Laptop (Windows, no NVIDIA GPU) = data prep + coding + report.
> University server (GPU, Linux) = building, training, rendering, timing.
> Anything marked **⚠ VERIFY** depends on repo version — check `--help` / README on your clone before relying on it.

---

## Table of Contents
0. [Big picture & fairness rules](#0-big-picture--fairness-rules)
1. [Laptop setup](#1-laptop-setup-windows)
2. [Inspect the video (ffprobe)](#2-inspect-the-video-with-ffprobe)
3. [Extract frames (ffmpeg)](#3-extract-frames-with-ffmpeg)
4. [Select sharp, well-spaced frames](#4-select-sharp-well-spaced-frames)
5. [Camera poses with COLMAP](#5-camera-poses-with-colmap)
6. [Build InstantNGP transforms.json](#6-build-instantngp-transformsjson)
7. [Train/test split](#7-traintest-split)
8. [Server access & file sync](#8-server-access--file-sync)
9. [Server environment: 3DGS](#9-server-environment-3d-gaussian-splatting)
10. [Server environment: InstantNGP](#10-server-environment-instantngp)
11. [Training both models](#11-training-both-models)
12. [Rendering test views](#12-rendering-test-views)
13. [Unified metrics (PSNR/SSIM/LPIPS)](#13-unified-metrics-psnrssimlpips)
14. [Rendering-time benchmark](#14-rendering-time-benchmark)
15. [Model size](#15-model-size)
16. [Figures](#16-figures)
17. [Report structure](#17-report-structure)
18. [Schedule](#18-schedule)
19. [Troubleshooting](#19-troubleshooting)
20. [Final checklist](#20-final-checklist)

---

## 0. Big picture & fairness rules

```
video ──ffmpeg──► frames ──blur filter──► input/ ──COLMAP──► images/ (undistorted) + sparse/0 (poses)
                                                               │
                              ┌────────────────────────────────┴───────────────┐
                              ▼                                                ▼
                  3DGS (reads COLMAP directly)               colmap2nerf.py ► transforms.json
                              │                                                │ split
                              │                                  transforms_train.json / _test.json
                              ▼                                                ▼
                     train.py (30k it)                          instant-ngp run.py (35k steps)
                              │                                                │
                      render test views                               render test views
                              └──────────────► SAME GT, SAME names, SAME res ◄─┘
                                                    │
                          eval_metrics.py (PSNR/SSIM/LPIPS) + timing + size + figures
```

**Fairness rules (write these into the report, too):**
1. ONE COLMAP reconstruction → both models use identical poses.
2. Both models train on the **undistorted** images in `data/scene/images/`.
3. Identical train/test split (same file names).
4. Identical resolution for training, rendering and evaluation.
5. One metric script for both models (don't mix each repo's own metric code).
6. Same GPU, idle, for all timing.

---

## 1. Laptop setup (Windows)

1. **Project folder layout** (create now; mirror it on the server):
   ```
   d:\assignment2\
     raw\fullvideo.mp4          ← move the video here
     data\candidates\           ← all extracted frames (before filtering)
     data\scene\input\          ← selected frames (COLMAP input)
     scripts\                   ← your own code
     logs\
     results\figures\
     report\
   ```
2. **Git**: `git init` in `d:\assignment2`, add a `.gitignore` excluding `raw/`, `data/`, `outputs/`, `*.ply`, `*.ingp`. Commit scripts only. (You can push to a private GitHub repo and `git pull` on the server — easier than copying scripts around.)
3. **ffmpeg**: install with ONE of
   - `winget install Gyan.FFmpeg` (then reopen terminal), or
   - `conda install -c conda-forge ffmpeg`
   Check: `ffmpeg -version` and `ffprobe -version`.
4. **Python env for prep scripts** (don't use your base Python 3.13 – some packages lag):
   `conda create -n prep python=3.10` → `conda activate prep` → `pip install opencv-python numpy matplotlib tqdm`
5. **COLMAP for Windows**: download the latest release zip from https://github.com/colmap/colmap/releases (the `nocuda` build is fine for your laptop). Unzip to e.g. `C:\tools\colmap\`. The launcher is `COLMAP.bat`. Test: `C:\tools\colmap\COLMAP.bat -h`.
   (Alternative: do COLMAP on the server – `conda install -c conda-forge colmap` – faster with GPU SIFT. Either is fine; just do it **once**.)
6. **Clone 3DGS on the laptop too** (only to use its `convert.py` wrapper for COLMAP):
   `git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive repos\gaussian-splatting`
7. **Clone instant-ngp on the laptop too** (only to use `scripts/colmap2nerf.py`; no build needed):
   `git clone --recursive https://github.com/nvlabs/instant-ngp repos\instant-ngp`

---

## 2. Inspect the video with ffprobe

Run these and **write the answers down** (they go in the report's capture section):

| What | Command |
|---|---|
| Resolution, codec, fps | `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,codec_name,nb_frames -of default=nw=1 raw/fullvideo.mp4` |
| Duration (s) | `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 raw/fullvideo.mp4` |
| Rotation (phones!) | `ffprobe -v error -select_streams v:0 -show_entries stream_side_data=rotation -of default=nw=1 raw/fullvideo.mp4` |

Notes:
- If a rotation (e.g. -90) is reported: ffmpeg **auto-rotates** on decode by default, so extracted frames will come out upright. Just open a couple of frames and confirm.
- Variable frame rate (common on phones) is fine for extraction with the `fps` filter.
- Watch the video once more, end to end, and note timestamps of anything bad (shake, someone walking by, a light flicker, a reflection). You will cut those out.

**Decide how many candidate frames you want.** Rule of thumb: extract ~3× more candidates than you'll keep (keep 60–120 → extract ~200–300).
`extract_fps = candidates_wanted / duration_seconds` (e.g. 250 / 60 s ≈ 4 fps).

---

## 3. Extract frames with ffmpeg

### 3.1 Choose output resolution
- 3DGS automatically downsizes training images wider than 1600 px. To keep both models on **exactly** the same resolution, resize yourself to **≤1600 px wide** (e.g. 1600×900 for 16:9, or 1280×720 if the server GPU is small).
- Keep the width/height **even** numbers.

### 3.2 Basic extraction (all candidates)
```
ffmpeg -i raw/fullvideo.mp4 -vf "fps=4,scale=1600:-2:flags=lanczos" -q:v 1 data/candidates/frame_%05d.jpg
```
Flag meanings:
- `-vf "fps=4"` – sample 4 frames per second of video, evenly.
- `scale=1600:-2` – width 1600, height auto, rounded to an even number; `flags=lanczos` = sharp resampling.
- `-q:v 1` – best JPEG quality (scale 1–31, lower = better). Use `-q:v 2` if disk space matters.
- `%05d` – zero-padded numbering → names sort correctly (important for the split!).

Lossless alternative (bigger files, no JPEG artefacts): replace `.jpg` with `.png` and drop `-q:v`.

### 3.3 Useful variations
- **Only a time range** (skip a bad start/end): put `-ss 00:00:03 -to 00:00:55` **before** `-i` for fast seeking:
  `ffmpeg -ss 3 -to 55 -i raw/fullvideo.mp4 -vf "fps=4,scale=1600:-2" -q:v 1 data/candidates/frame_%05d.jpg`
- **Every Nth frame instead of fps** (e.g. every 8th decoded frame):
  `-vf "select='not(mod(n\,8))',scale=1600:-2" -vsync vfr`
- **Drop near-duplicate frames** (camera paused): add `mpdecimate` before scale: `-vf "mpdecimate,fps=4,scale=1600:-2"`.
- **Only keyframes** (often sharper on some codecs): `-skip_frame nokey -i raw/fullvideo.mp4 -vsync vfr ...`.
- **Contact sheet for a quick visual check** (one big image of thumbnails):
  `ffmpeg -i raw/fullvideo.mp4 -vf "fps=1,scale=320:-2,tile=6x6" -frames:v 1 results/figures/contact_sheet.jpg`
  (Nice figure for the report's dataset section, too.)

### 3.4 Check
- Count: `(Get-ChildItem data\candidates).Count` (PowerShell).
- Open 5–10 random frames at 100% zoom: sharp? upright? no compression blocking?

---

## 4. Select sharp, well-spaced frames

### 4.1 Script to write: `scripts/filter_frames.py`
**Inputs:** `data/candidates/`, args `--window W` (e.g. 3), `--min_sharp T`, `--out data/scene/input`.
**Steps to implement:**
1. List files, **sorted** by name.
2. For each frame: read in grayscale with OpenCV, compute sharpness = variance of `cv2.Laplacian(img, cv2.CV_64F)`.
3. Save all scores to `results/sharpness.csv` (name, score).
4. Split the sorted list into consecutive windows of W frames; in each window keep the frame with the highest score (this keeps spacing even and removes the blurriest frames).
5. Drop kept frames whose score < T (choose T by looking at the histogram — the left tail is blurry).
6. Copy survivors to `data/scene/input/`, **renaming** them sequentially (`0000.jpg`, `0001.jpg`, …) so sorted order = capture order.
7. Plot score vs frame index with the kept frames highlighted → `results/figures/sharpness.png` (report figure).

### 4.2 Manual pass (don't skip!)
Go through `data/scene/input/` and delete frames with: motion blur, moving objects/people, visible shadows of you/the camera, strong reflections, exposure jumps. The assignment explicitly lists these.

### 4.3 Target
- **60–120 frames**, adjacent frames overlapping ~70–80% (the same object visible in many consecutive frames).
- If you end up < 40 frames → lower the window W or extract at higher fps.

**From here on, freeze `data/scene/input/`.** Changing it later = redo COLMAP + everything.

---

## 5. Camera poses with COLMAP

### 5.1 Run via 3DGS's `convert.py` (recommended)
It runs: feature extraction → exhaustive matching → mapping → **undistortion** → produces exactly what 3DGS wants.

Laptop (Windows, CPU):
```
conda activate prep
python repos\gaussian-splatting\convert.py -s data\scene --colmap_executable "C:\tools\colmap\COLMAP.bat" --no_gpu
```
Server (Linux, GPU): `python repos/gaussian-splatting/convert.py -s data/scene`

**⚠ VERIFY** flags with `python convert.py -h` (`--no_gpu`, `--skip_matching`, `--camera`, `--colmap_executable`, `--resize` exist in current versions). Default camera model is `OPENCV`; images are shared as a single camera — correct for one video.

Expected output:
```
data/scene/
  input/            your frames (untouched)
  distorted/        raw COLMAP db + sparse
  images/           UNDISTORTED images  ← ground truth for BOTH models
  sparse/0/         cameras.bin images.bin points3D.bin (PINHOLE)
  stereo/ ...       (ignore)
```
Time: ~100 images on a laptop CPU, exhaustive matching ≈ 10–40 min.

### 5.2 Validate the reconstruction (IMPORTANT)
1. Open COLMAP GUI (`COLMAP.bat gui`) → File → Import model → choose `data/scene/sparse/0` (set image path to `data/scene/images`).
2. Check:
   - **Number of registered images == number of input images.** (Shown in the GUI status bar, or run `COLMAP.bat model_analyzer --path data/scene/sparse/0` → "Registered images", "Mean reprojection error" (should be < ~1 px), "Points".)
   - Cameras form a smooth horizontal line (your capture); no cameras flipped or far away.
   - Point cloud looks like your scene.
3. Note the numbers for the report (registered images, #points, mean reprojection error).
4. Note: undistortion slightly crops images → `images/` may be a little smaller than `input/`. That's fine; `images/` is the GT resolution from now on.

### 5.3 If COLMAP fails / registers few images
- Too little overlap → re-select frames with smaller window (more frames).
- Textureless scene (white walls) → choose a more textured scene; nothing else fixes this well.
- Split into two models → run with sequential matching instead: run COLMAP manually (`feature_extractor`, `sequential_matcher`, `mapper`, `image_undistorter`) mirroring the commands printed by `convert.py`.

### 5.4 Convert the model to text (for InstantNGP)
```
COLMAP.bat model_converter --input_path data\scene\sparse\0 --output_path data\scene\colmap_text --output_type TXT
```
(create `data\scene\colmap_text` first). You should get `cameras.txt`, `images.txt`, `points3D.txt`. Check `cameras.txt` says `PINHOLE`.

---

## 6. Build InstantNGP transforms.json

```
python repos\instant-ngp\scripts\colmap2nerf.py --images data\scene\images --text data\scene\colmap_text --out data\scene\transforms.json --aabb_scale 16
```
- **Do NOT use `--run_colmap`** – we reuse the same reconstruction as 3DGS.
- `--aabb_scale`: power of 2 (1–128). Scenes extending beyond the camera region need larger values; 16 is a good start for real scenes; try 4/8/32 if quality is bad, but report the value you used.
- **⚠ VERIFY** `python colmap2nerf.py -h` for your clone's flags.

**Check `transforms.json`:**
- Top-level: `camera_angle_x`, `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`, `aabb_scale`, `frames`.
- `w`/`h` equal the size of images in `data/scene/images/`.
- Each frame has `file_path` and a 4×4 `transform_matrix`; count frames == registered images.
- `file_path` is relative to the json's folder — check the paths resolve (e.g. `images/0000.jpg` or `./images/0000.jpg`). If they contain Windows backslashes or absolute laptop paths, fix them (your split script can rewrite them) because the server is Linux.
- colmap2nerf re-centres and re-scales the poses — that's expected; it doesn't affect the comparison since the images and split are identical.

---

## 7. Train/test split

### 7.1 Rule
Use **every 8th image** (K = 8) as test, like 3DGS's `--eval` mode: in the **sorted list of image names**, image index `i` is test if `i % 8 == 0`.
**⚠ VERIFY** in `repos/gaussian-splatting/scene/dataset_readers.py` → function `readColmapSceneInfo`: look for `llffhold` and how test cameras are chosen (newer versions may read a `test.txt` from the sparse folder; older ones use `idx % llffhold == 0` on sorted names). Your NGP split must reproduce that exact rule.

### 7.2 Script to write: `scripts/make_split.py`
**Inputs:** `data/scene/transforms.json`, `--k 8`. **Outputs:** `transforms_train.json`, `transforms_test.json`, `results/test_list.txt`, `results/train_list.txt`.
**Steps:**
1. Load json; collect frames; sort by the **basename** of `file_path`.
2. Enumerate; `i % k == 0` → test, else train.
3. Write two jsons that copy **all** top-level keys (intrinsics, `aabb_scale`, etc.) and only differ in `frames`.
4. Normalise `file_path` to forward slashes, relative path.
5. Print counts; assert train ∩ test = ∅ and train ∪ test = all.
6. Write names to the two txt lists (used later for evaluation and to cross-check with 3DGS).

Expected with 96 images: 12 test, 84 train.

**Cross-check later:** after 3DGS `render.py`, the names in `outputs/gs/.../test/ours_30000/gt/` correspond to the test cameras — 3DGS renames them `00000.png, 00001.png...` in order, so compare ordering against `test_list.txt` (they should be the same sorted order). Your metric script maps by order + list.

---

## 8. Server access & file sync

1. **Connect:** `ssh user@server` (on campus or via university VPN). Set up SSH keys (`ssh-keygen`, `ssh-copy-id` or append to `~/.ssh/authorized_keys`) so sshfs/rsync don't ask passwords.
2. **sshfs on Windows** = install **WinFsp** + **SSHFS-Win** (github.com/winfsp/sshfs-win); map drive via Explorer: `\\sshfs\user@server\path`. Use it to **edit code** in VS Code. (Alternatively VS Code "Remote-SSH" extension – often smoother than sshfs.)
3. **Move data with scp/rsync, not sshfs** (much faster, and avoid training reading through sshfs):
   `scp -r d:\assignment2\data\scene user@server:~/assignment2/data/`
   (or `rsync -avP` from Git Bash/WSL).
4. **Always run jobs in `tmux`** so they survive disconnects: `tmux new -s a2`, detach `Ctrl-b d`, reattach `tmux attach -t a2`.
5. **Check the GPU before anything:**
   - `nvidia-smi` → GPU model, VRAM, driver, max CUDA, **who else is using it**.
   - `nvcc --version` (CUDA toolkit), `gcc --version`, `cmake --version`, `conda --version`.
   - Compute capability: `nvidia-smi --query-gpu=name,compute_cap --format=csv` (e.g. 8.6 for RTX 30xx / A6000, 8.0 for A100, 8.9 for RTX 40xx/L40, 7.5 for T4/RTX 20xx).
   - If several GPUs: pick one with `export CUDA_VISIBLE_DEVICES=0` and use the same one for everything.
   - If cluster uses SLURM: you'll submit jobs with `sbatch`/`srun --gres=gpu:1` — ask TA/admin.
6. If no conda: install Miniconda in your home dir (`bash Miniconda3-latest-Linux-x86_64.sh`).
7. Record versions now (`nvidia-smi`, `nvcc`, `gcc`, python, torch) into `logs/env.txt` → report.

---

## 9. Server environment: 3D Gaussian Splatting

Repo: https://github.com/graphdeco-inria/gaussian-splatting

1. `git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive repos/gaussian-splatting`
   If you forgot `--recursive`: `git submodule update --init --recursive`.
   Record commit: `git -C repos/gaussian-splatting rev-parse HEAD`.
2. **Option A – official env:** `conda env create --file environment.yml` → `conda activate gaussian_splatting`. Works if the server has a compatible toolkit (old CUDA 11.x pins). Often fails on new systems → Option B.
3. **Option B – manual env (usually works):**
   1. `conda create -n gs python=3.10 -y && conda activate gs`
   2. Get a CUDA toolkit matching the PyTorch build you'll install. E.g. CUDA 11.8:
      `conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y` (or `module load cuda/11.8`). Check `which nvcc` → inside env.
   3. PyTorch built for that CUDA (get exact command from pytorch.org "previous versions"), e.g. `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118`
   4. Check: `python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` → True.
   5. `pip install plyfile tqdm opencv-python joblib`
   6. Set arch so extensions compile for your GPU: `export TORCH_CUDA_ARCH_LIST="8.6"` (your compute cap).
   7. `pip install submodules/diff-gaussian-rasterization submodules/simple-knn` and `submodules/fused-ssim` **if that folder exists** in your clone (⚠ VERIFY `ls submodules`). Use `--no-build-isolation` if pip complains it can't find torch.
   8. Check: `python -c "import diff_gaussian_rasterization, simple_knn"`.
4. **Smoke test** (1000 iterations, ~1 min):
   `python train.py -s ~/assignment2/data/scene -m ~/assignment2/outputs/gs/smoke --eval --iterations 1000`
   The log should say how many train / test cameras it loaded → **must match your split counts**.
5. Note: train.py opens a network-viewer port (6009). On a shared server, if it errors "address in use", pass `--port 60xx`.

---

## 10. Server environment: InstantNGP

Repo: https://github.com/NVlabs/instant-ngp

1. Requirements: CUDA ≥ 10.2 (11.x/12.x fine), **cmake ≥ 3.21**, a gcc supported by your CUDA (CUDA 11.x → gcc ≤ 11; CUDA 12.x → gcc ≤ 12/13), Python 3.7+.
2. `conda create -n ngp python=3.10 -y && conda activate ngp`
   `pip install cmake` (gets a new cmake if system one is old) — check `cmake --version`.
   If needed, same CUDA toolkit trick as §9 (`conda install ... cuda-toolkit`), and if gcc is too new: `conda install -c conda-forge gxx=11` and `export CC=$(which gcc) CXX=$(which g++)` — or `module load gcc/11`.
3. `git clone --recursive https://github.com/nvlabs/instant-ngp repos/instant-ngp && cd repos/instant-ngp`; record commit hash.
4. `pip install -r requirements.txt`
5. Configure (headless; no GUI on server):
   `cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_GUI=off`
   If the GPU arch isn't detected (or you build on a login node without GPU): `export TCNN_CUDA_ARCHITECTURES=86` (your compute cap ×10, no dot) before cmake.
6. Build (10–30 min): `cmake --build build --config RelWithDebInfo -j 8`
7. Check python bindings: from `repos/instant-ngp`, `python -c "import sys; sys.path.append('build'); import pyngp as ngp; print('ok')"`.
8. **Smoke test:** `python scripts/run.py --scene ~/assignment2/data/scene/transforms_train.json --n_steps 1000` → prints loss and finishes.
   **⚠ VERIFY** `python scripts/run.py -h` for: `--scene`, `--n_steps`, `--save_snapshot`, `--load_snapshot`, `--test_transforms`, `--screenshot_transforms`, `--screenshot_frames`, `--screenshot_dir`, `--screenshot_spp`, `--width`, `--height`.
9. **Fallback** if the build is impossible: nerfstudio (`pip install nerfstudio`, method `instant-ngp`). It's a re-implementation (nerfacc), so state that clearly in the report.

---

## 11. Training both models

General: run each inside `tmux`, GPU otherwise idle, log stdout, time everything.

### 11.1 3DGS
```
/usr/bin/time -v python train.py \
  -s ~/assignment2/data/scene -m ~/assignment2/outputs/gs/scene \
  --eval -r 1 --iterations 30000 \
  --test_iterations 1000 3000 7000 15000 20000 30000 \
  --save_iterations 7000 30000 \
  2>&1 | tee ~/assignment2/logs/gs_train.log
```
- `--eval` → holds out test views; `-r 1` → full resolution (you already resized to ≤1600).
- `--test_iterations` → it prints test L1/PSNR at each → **your convergence curve for 3DGS** (grep them from the log).
- `/usr/bin/time -v` → "Elapsed (wall clock) time" = **training time** and "Maximum resident set size". Note: this includes data loading; that's fine, report it as wall-clock. Also note peak GPU memory by watching `nvidia-smi` (or `watch -n 5 nvidia-smi` in another tmux pane).
- 30k iterations is the paper's standard; typical time 15–40 min.
- Outputs: `outputs/gs/scene/point_cloud/iteration_30000/point_cloud.ply`, `cfg_args`, `cameras.json`.

### 11.2 InstantNGP
```
/usr/bin/time -v python scripts/run.py \
  --scene ~/assignment2/data/scene/transforms_train.json \
  --n_steps 35000 \
  --save_snapshot ~/assignment2/outputs/ngp/scene_35k.ingp \
  --test_transforms ~/assignment2/data/scene/transforms_test.json \
  2>&1 | tee ~/assignment2/logs/ngp_train.log
```
- 35k steps is the commonly used "converged" setting (default in run.py when n_steps unset — ⚠ VERIFY).
- `--test_transforms` → run.py prints its own PSNR/SSIM at the end (use only as a sanity check; final numbers come from §13).
- Wall clock from `time -v` includes the evaluation pass; to get pure training time either run without `--test_transforms`, or read the training-loop time from the log.
- **Convergence curve:** also train/save at 1k, 3k, 7k, 15k, 20k (separate runs with different `--n_steps`, or load snapshot and continue with `--load_snapshot` + more steps), evaluate each with §12–13 → PSNR vs steps.

### 11.3 (Optional, extra credit-ish) Hyper-parameter notes
- 3DGS: also try `--iterations 7000` to show the speed/quality tradeoff.
- NGP: try `aabb_scale` 4/16/32; mention best.
- Keep the **main** comparison at default, converged settings.

---

## 12. Rendering test views

Goal: three folders with **identical file names and identical resolution**:
```
outputs/eval/gt/    0000.png ...   (copied from data/scene/images for test names)
outputs/eval/gs/    0000.png ...
outputs/eval/ngp/   0000.png ...
```

### 12.1 3DGS
`python render.py -m ~/assignment2/outputs/gs/scene --skip_train` (activate `gs` env).
Output: `outputs/gs/scene/test/ours_30000/renders/00000.png ...` and `.../gt/00000.png ...`.
Then map them (by order) to the names in `results/test_list.txt` and copy into `outputs/eval/gs/`. Visually compare the 3DGS `gt/` image with your GT of the same name to confirm the mapping.

### 12.2 InstantNGP — script to write: `scripts/render_ngp.py` (run in `ngp` env)
Simplest path: read `scripts/run.py` from the repo — its `--screenshot_transforms` and `--test_transforms` sections already show the exact pyngp calls. Steps:
1. Add `repos/instant-ngp/build` to `sys.path`, `import pyngp as ngp`.
2. Create `ngp.Testbed()`, load the training scene (or just `load_snapshot(scene_35k.ingp)` — check run.py for how it sets the mode for NeRF).
3. Read `transforms_test.json`. For each frame:
   - set camera from the frame's `transform_matrix` (run.py uses `set_nerf_camera_matrix` with the top 3×4 rows),
   - set FOV from `camera_angle_x` (and principal point if your version supports it — check run.py),
   - render at **w × h of the GT**, fixed `spp` (e.g. 8 for quality images; note it),
   - convert linear → sRGB the way run.py does, drop/composite alpha onto the same background as GT (black is common — be consistent),
   - save PNG with the **same basename** into `outputs/eval/ngp/`.
4. Disable training while rendering (`testbed.shall_train = False`).
5. Copy GT test images into `outputs/eval/gt/` (convert to PNG, same name).
6. Sanity: open GT vs render side by side; they must be pixel-aligned (no shift/flip). If misaligned → camera convention/FOV/principal point problem.

---

## 13. Unified metrics (PSNR/SSIM/LPIPS)

Script to write: `scripts/eval_metrics.py` (server, `gs` env has torch; `pip install lpips torchmetrics` or `pytorch-msssim`).
**Inputs:** `--gt outputs/eval/gt --pred outputs/eval/gs` (run once per method), `--out results/metrics_<method>.csv`.
**Steps:**
1. Iterate names from `test_list.txt`; load GT & pred as RGB float32 in [0,1], tensor shape (1,3,H,W) on GPU. Assert same shape.
2. **PSNR** = 10·log10(1 / MSE).
3. **SSIM** – standard 11×11 Gaussian window (torchmetrics `StructuralSimilarityIndexMeasure(data_range=1.0)`).
4. **LPIPS** – `lpips.LPIPS(net='vgg')` (3DGS paper uses VGG); input scaled to [-1,1] (`normalize=True` option handles it). Optionally also report AlexNet.
5. Per-image rows + mean row → CSV. Also print mean ± std.
6. **Validation:** run your script on 3DGS's own `renders/` vs `gt/` and compare with `python metrics.py -m outputs/gs/scene` from the repo — should agree closely. If not, debug before trusting NGP numbers.
7. Per-image results let you plot PSNR per test view (which views are hard for each method).

---

## 14. Rendering-time benchmark

Measure **pure render time**, excluding disk I/O, same GPU, same resolution, GPU idle.

### 14.1 3DGS — `scripts/time_gs.py` (gs env)
Base it on `render.py` (load `ModelParams`/`PipelineParams`, `GaussianModel`, `Scene(..., load_iteration=30000, shuffle=False)`).
1. Get test cameras, background color as in render.py.
2. Warm-up: render 10 times (not timed).
3. For each test cam, repeat R=10 times: `torch.cuda.synchronize()` → t0 → `render(...)` → `torch.cuda.synchronize()` → t1.
4. Report mean ms/frame, FPS = 1000/ms, at H×W.

### 14.2 InstantNGP — `scripts/time_ngp.py` (ngp env)
Same structure: load snapshot, set camera per test frame, warm-up, time `testbed.render(w, h, spp, True)` calls (render returns after the GPU finishes since it copies the image back to host → timing is valid but includes the device→host copy; mention it).
- Report for **spp=1** (fair "one sample per pixel" speed) and optionally the spp used for quality images.

### 14.3 Report
Table: method, resolution, spp, ms/frame, FPS, GPU model. Expect 3DGS ≫ NGP in FPS (rasterization vs. ray-marching).

---

## 15. Model size

- **3DGS:** `ls -lh outputs/gs/scene/point_cloud/iteration_30000/point_cloud.ply`. Also number of Gaussians: read the PLY header (`head -c 2000 point_cloud.ply` shows `element vertex N`) — each Gaussian stores 59 floats (pos 3, normals 3 (unused), SH DC 3 + rest 45, opacity 1, scale 3, rot 4) ≈ 248 bytes.
- **InstantNGP:** `ls -lh outputs/ngp/scene_35k.ingp`. ⚠ VERIFY whether your version saves compressed (`.ingp` is msgpack, compressed by default in recent versions). Optionally: theoretical params = hash table size (2^19 default log2_hashmap_size) × feature dim (2) × levels (16) + small MLPs, stored in fp16 → explain why it's much smaller.
- Also report: peak GPU memory during training (from monitoring), and number of input images / resolution.

---

## 16. Figures

Script to write: `scripts/make_figures.py` (matplotlib + numpy + PIL).
1. **Qualitative grid:** 3–4 test views × [GT | InstantNGP | 3DGS], with PSNR printed under each image.
2. **Zoom crops:** pick a detail region (text, edges, thin structures, textures) — draw a box on full image, show the crop enlarged for all three.
3. **Error maps:** |GT − pred| averaged over channels, shown with a colormap (same color scale for both methods).
4. **Convergence plot:** test PSNR vs iterations for both methods (from §11 logs / snapshots).
5. **Bar charts:** training time, FPS (log scale), model size (MB, log scale).
6. **Per-view PSNR plot:** x = test view index along the camera path, y = PSNR, one line per method (views at the ends of the trajectory are usually worst — discuss).
7. **Dataset figures:** contact sheet of input frames, COLMAP camera/point-cloud screenshot, sharpness plot.
8. (Optional) fly-through video from each method: render a smooth path (e.g. interpolate between train poses) and encode with `ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4`; side-by-side: `ffmpeg -i ngp.mp4 -i gs.mp4 -filter_complex hstack sbs.mp4`.

---

## 17. Report structure

1. **Introduction** – task, the two methods in 2–3 lines each (NGP: multiresolution hash encoding + tiny MLP + ray-marching volume rendering; 3DGS: explicit anisotropic Gaussians, tile-based differentiable rasterization).
2. **Data capture** – device, scene description, capture motion (horizontal slide), how blur/motion/shadow/reflection were avoided, video specs (from §2), extraction settings (ffmpeg command), selection method + counts, final resolution, contact sheet.
3. **Pose estimation** – COLMAP settings, registered images, reprojection error, #points, screenshot.
4. **Split** – K=8, #train/#test, list of test images.
5. **Setup** – hardware (GPU, CPU, RAM), software versions, repo commits, hyper-parameters (iterations, aabb_scale, spp, resolution, background), timing methodology.
6. **Results** – main table:
   | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Train time | Render ms / FPS | Model size | #Gaussians / params |
   + convergence plot + qualitative figures + per-view plot.
7. **Discussion** – quality differences (floaters, blur, view extrapolation at trajectory ends, sparse-region behavior), speed (why), size (why), training time, limitations of a 1-D horizontal capture (little vertical parallax → poor novel views off-path), what you'd do differently.
8. **Conclusion** + references (Müller et al. 2022 SIGGRAPH; Kerbl et al. 2023 SIGGRAPH; COLMAP – Schönberger & Frahm 2016).
9. Appendix: exact commands used.

---

## 18. Schedule (today = Oct 19)

| Day | Tasks |
|---|---|
| **Oct 19 (laptop)** | §1–4: setup, ffprobe, extract, filter, manual pass. Start COLMAP (§5) and let it run. |
| **Oct 20 (laptop → server)** | §5–7: validate COLMAP, transforms.json, split. Copy data to server. Start §9–10 env setup (biggest risk – start early!). |
| **Oct 21 (server)** | Finish envs, smoke tests, full training runs (§11). Write render/eval/timing scripts meanwhile. |
| **Oct 22** | §12–15: renders, metrics (validate vs 3DGS metrics.py), timing, size. Convergence snapshots. |
| **Oct 23** | §16 figures; start report. |
| **Oct 24** | Finish report; re-run anything suspicious. |
| **Oct 25** | Buffer + submit. |

---

## 19. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| COLMAP registers only part of the images | Not enough overlap / blur → more frames, smaller window; sequential matcher; better texture |
| `convert.py` can't find colmap on Windows | pass `--colmap_executable "C:\tools\colmap\COLMAP.bat"` |
| 3DGS submodule build: `nvcc not found` / CUDA mismatch | toolkit version must match `torch.version.cuda`; install toolkit in conda env; set `CUDA_HOME=$CONDA_PREFIX` |
| 3DGS build: unsupported gcc | install `gxx=11` from conda-forge and export CC/CXX |
| 3DGS: `no kernel image is available` | wrong arch → set `TORCH_CUDA_ARCH_LIST` and reinstall submodules (`pip install --force-reinstall --no-build-isolation ...`) |
| 3DGS CUDA OOM | lower resolution (`-r 2`) for **both** methods, or `--data_device cpu` |
| NGP cmake too old | `pip install cmake` in env |
| NGP `TCNN` arch errors / built on login node | set `TCNN_CUDA_ARCHITECTURES` explicitly |
| NGP render shifted / mirrored vs GT | camera convention or FOV/principal point not set like run.py does; re-check with run.py's screenshot code |
| NGP very blurry / background mush | try different `aabb_scale`; check poses (transforms paths & count) |
| NGP colors washed out vs GT | missing linear→sRGB conversion |
| Your metrics ≠ 3DGS metrics.py | data range / channel order / LPIPS normalisation bug |
| sshfs very slow | edit only over sshfs; move data via scp/rsync; run on server disk |
| Shared GPU busy | timings invalid → wait, or pin `CUDA_VISIBLE_DEVICES` to a free GPU and note it |

---

## 20. Final checklist

- [ ] Own capture only; capture conditions documented
- [ ] Same COLMAP poses, same undistorted images, same split, same resolution for both models
- [ ] 3DGS trained 30k iterations, NGP 35k steps; convergence plot shows plateau
- [ ] PSNR / SSIM / LPIPS from ONE script; validated against 3DGS metrics.py
- [ ] Render time measured with sync + warm-up, same GPU, stated resolution/spp
- [ ] Model sizes (+ #Gaussians / params)
- [ ] Training wall-clock times, same GPU
- [ ] Qualitative grids, zoom crops, error maps
- [ ] Software versions + commit hashes + hardware in report
- [ ] All exact commands in appendix
