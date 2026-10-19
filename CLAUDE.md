# Handoff: InstantNGP vs 3D Gaussian Splatting (GPU run on a Windows laptop)

You are running the GPU part of a course assignment on this machine. Data preparation and
all pipeline code are DONE and VERIFIED. Your job: set up WSL2, clone and build the two model
repos, run the pipeline, check the outputs, report results. Do not redesign anything.
Follow the steps in order; each step ends with a check that must pass before continuing.

## The assignment (context)

Compare InstantNGP and 3DGS on a self-captured static scene: PSNR / SSIM / LPIPS + visual
inspection on held-out test views, rendering time, model size, training time. Both models
trained to convergence. All timings must come from ONE GPU (this laptop's).

## What is already in this repo (do not modify)

| Path | Content |
|---|---|
| `data/scene/images/` | 62 undistorted frames, 701x1249 (ground truth for BOTH models) |
| `data/scene/input/` | original frames (evidence of own capture; not used for training) |
| `data/scene/sparse/0/` | COLMAP 4.2 model, PINHOLE, 62/62 registered, 16,573 points, 0.40 px error (3DGS reads this) |
| `data/scene/colmap_text/` | same model as text (used by scripts) |
| `data/scene/transforms_train.json`, `transforms_test.json` | InstantNGP poses, recentred (pitfall 3), aabb_scale 4; 54 train / 8 test |
| `data/scene/novel/*.json` | 7 novel camera paths, identical cameras for both models (verified) |
| `results/test_list.txt`, `train_list.txt` | split: sorted names, index % 8 == 0 -> test (= 3DGS `--eval`) |
| `scripts/` | the pipeline (`scripts/README.md`) |

Do NOT re-run COLMAP, `prepare_ngp.py` or `make_novel_paths.py`, and do not edit `data/`.

## Why WSL2

3DGS and InstantNGP build cleanly on Linux; native Windows builds need Visual Studio and are
fragile, and the run scripts are bash. WSL2 runs Ubuntu on Windows with full CUDA access
through the normal Windows NVIDIA driver. Use **Ubuntu 22.04** specifically: its gcc 11 is
supported by CUDA 11.8 (Ubuntu 24.04 ships gcc 13, which CUDA 11.8 rejects).

Two terminals are used below:
- **[PowerShell]** = Windows PowerShell (Start menu -> "PowerShell", run as Administrator where noted)
- **[Ubuntu]** = the Ubuntu (WSL) terminal

---

## Step 1 - Windows checks  [PowerShell]

```powershell
nvidia-smi
```
Must show an NVIDIA GPU and a driver. If the command is missing or old, install the latest
Game Ready / Studio driver from nvidia.com and reboot. Do NOT install any NVIDIA driver inside Ubuntu later.

Note the GPU name. Look up its compute capability: RTX 20xx = 7.5, RTX 30xx = 8.6,
RTX 40xx = 8.9 (laptop variants are the same). Called `ARCH` below (e.g. 86 / 8.6).

Plug the laptop into power and set Windows power mode to "Best performance" (battery mode
throttles the GPU and ruins the timings).

## Step 2 - install WSL2 + Ubuntu 22.04  [PowerShell as Administrator]

```powershell
wsl --install -d Ubuntu-22.04
```
Reboot if asked. On first start Ubuntu asks for a Linux username and password (remember it, `sudo` needs it).
If WSL is already installed: `wsl --update` then `wsl --install -d Ubuntu-22.04`.
Check: `wsl -l -v` shows `Ubuntu-22.04` with VERSION `2`.

Optional but recommended if the laptop has 16 GB RAM or less: create `C:\Users\<you>\.wslconfig` with
```
[wsl2]
memory=12GB
swap=16GB
```
then `wsl --shutdown` and reopen Ubuntu.

## Step 3 - Ubuntu base packages  [Ubuntu]

```bash
nvidia-smi                       # must show the same GPU as in Windows
sudo apt update
sudo apt install -y build-essential git tmux ffmpeg wget unzip \
    libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
gcc --version                    # must be 11.x
```

Install Miniconda:
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash
conda --version
```

## Step 4 - get this assignment repo into Ubuntu  [Ubuntu]

Work inside the Linux home folder, NOT under `/mnt/c/...` (10x slower, breaks timings).
```bash
cd ~
git clone https://github.com/<OWNER>/<REPO>.git assignment2     # the repo this file is in
cd ~/assignment2
ls data/scene/images | wc -l                                   # must print 62
```
The repo is private: when git asks for a password, use a GitHub personal access token
(github.com -> Settings -> Developer settings -> Personal access tokens), or run
`sudo apt install -y gh && gh auth login` first.

Save the environment for the report:
```bash
mkdir -p logs
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv; gcc --version | head -1; uname -a; } > logs/env.txt
```

## Step 5 - clone the two model repos at the pinned commits  [Ubuntu, in ~/assignment2]

The scripts were written and checked against these exact commits. Do not use newer ones.
```bash
mkdir -p repos
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting repos/gaussian-splatting
git -C repos/gaussian-splatting checkout 54c035f7834b564019656c3e3fcc3646292f727d
git -C repos/gaussian-splatting submodule update --init --recursive

git clone --recursive https://github.com/NVlabs/instant-ngp repos/instant-ngp
git -C repos/instant-ngp checkout abe236ee00cf90cfca6e36e65c00435d5b21f50a
git -C repos/instant-ngp submodule update --init --recursive
```
Check: `ls repos/gaussian-splatting/submodules` lists `diff-gaussian-rasterization fused-ssim simple-knn`,
and `ls repos/instant-ngp/dependencies/tiny-cuda-nn` is not empty.
(`scripts/patches/gs_convert_colmap4.patch` is only for re-running COLMAP - not needed.)

## Step 6 - 3DGS environment `gs`  [Ubuntu, in ~/assignment2]

Replace `8.6` with your ARCH from step 1.
```bash
conda create -n gs python=3.10 -y
conda activate gs
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True` and the GPU name. Then:
```bash
pip install plyfile tqdm opencv-python joblib lpips matplotlib scipy pillow
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"
pip install --no-build-isolation repos/gaussian-splatting/submodules/diff-gaussian-rasterization \
    repos/gaussian-splatting/submodules/simple-knn repos/gaussian-splatting/submodules/fused-ssim
python -c "import diff_gaussian_rasterization, simple_knn; print('3DGS extensions OK')"
```
If `nvcc`/`which nvcc` is not inside `$CONDA_PREFIX`, run `conda activate gs` again.

Smoke test (about 1 minute):
```bash
python repos/gaussian-splatting/train.py -s data/scene -m outputs/gs_smoke --eval -r 1 --iterations 500 --disable_viewer
```
Check: the log shows `LLFF HOLD`, finishes with `Training complete.`, and
`ls outputs/gs_smoke/point_cloud/iteration_500` shows `point_cloud.ply`. Then `rm -rf outputs/gs_smoke`.

## Step 7 - InstantNGP environment `ngp`  [Ubuntu, in ~/assignment2]

Replace `86` with your ARCH without the dot. The build takes 10-30 minutes.
```bash
conda deactivate
conda create -n ngp python=3.10 -y
conda activate ngp
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install cmake
pip install -r repos/instant-ngp/requirements.txt
export CUDA_HOME=$CONDA_PREFIX
export TCNN_CUDA_ARCHITECTURES=86
cd repos/instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_GUI=off -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc
cmake --build build --config RelWithDebInfo -j 4
cd ~/assignment2
python -c "import sys; sys.path.insert(0,'repos/instant-ngp/scripts'); import common, pyngp; print('pyngp OK')"
```
If the build runs out of memory (WSL killed / "Killed signal"), rerun the build line with `-j 2`.

Smoke test:
```bash
python scripts/train_ngp.py --scene data/scene/transforms_train.json --out_dir outputs/ngp_smoke --checkpoints 500
python scripts/render_ngp.py --snapshot outputs/ngp_smoke/step_000500.ingp --transforms data/scene/transforms_test.json --out outputs/ngp_smoke/test
ls outputs/ngp_smoke/test        # 8 PNGs: 0000.png ... 0056.png
```
The images should look like a slightly blurry shelf (only 500 steps). Then `rm -rf outputs/ngp_smoke`.

## Step 8 - run the full pipeline  [Ubuntu, in ~/assignment2]

Env names and iteration lists are in `scripts/config.sh` (`gs` / `ngp`; 3DGS 30k iterations,
InstantNGP 35k steps, checkpoints 1k/3k/7k/15k/final). The scripts activate the envs themselves.
```bash
conda deactivate
DRY_RUN=1 bash scripts/run_all.sh        # prints every command, runs nothing
tmux new -s a2                           # keeps running if the terminal closes (detach: Ctrl-b d)
bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
```
Before starting: close games, video, other GPU apps; laptop plugged in; disable sleep
(Windows Settings -> Power -> Screen and sleep -> Never while plugged in).
Expected time on an RTX 3090-class GPU: ~30-50 min; laptop GPUs may take 1.5-2x longer.
If one stage fails, fix it and rerun only that stage: `bash scripts/run_gs.sh`,
`bash scripts/run_ngp.sh`, or `bash scripts/run_eval.sh`.

## Step 9 - verify before reporting

1. `results/summary.md` exists; every cell has a number. Exception: "Peak GPU memory,
   process" is usually "–" under WSL2 (its nvidia-smi does not list processes); the
   whole-GPU memory row is then used, and the figure says so.
2. 3DGS cross-check line in `summary.md`: official metrics.py PSNR and SSIM must match ours
   within ~0.01 (LPIPS differs on purpose, see pitfall 5).
3. `results/figures/` has 22 PNGs (01-15); `results/videos/` has 7 MP4s. Open
   `01_test_views_a.png` and `12_novel_overview.png` (from Windows: `explorer.exe results/figures`):
   renders must look like the shelf and line up with ground truth. A shifted or mirrored
   render means a pose bug - stop and report it.
4. `05_convergence.png`: PSNR flattens over the last checkpoints for both methods.
5. Sanity range for this scene: PSNR about 22-32 dB, SSIM 0.7-0.95. PSNR near 10 dB or
   SSIM < 0.4 means something is wrong (split, poses, or a model reset).
6. Logs of `render_gs.py` print `pose check: max |projection difference| ... = <small>` (< 1e-4).

## Step 10 - hand back

Report to the user: the `results/summary.md` tables, GPU name, any warnings, anything changed.
Commit (do NOT push unless the user says so): `results/`, `logs/*.json`, `logs/*.txt`, `logs/*.log`
(use `git add -f logs/...` since `logs/` is gitignored). Do NOT commit `outputs/` (large);
if the user wants renders, zip them: `zip -r renders.zip outputs/eval outputs/novel`.
To copy files to Windows: `cp renders.zip /mnt/c/Users/<you>/Desktop/`.

## Known pitfalls (already handled in the code - do not "fix")

1. InstantNGP's loader sorts frames by file_path and drops frames whose image is missing;
   `render_ngp.py` stages each frame as a uniquely named copy of a real image on purpose.
2. InstantNGP resets the network if a json's `aabb_scale` differs from training;
   `render_ngp.py` refuses to run in that case.
3. `transforms*.json` were recentred because colmap2nerf.py's "centre of attention" fails for
   a sideways (forward-facing) capture; it is a similarity transform only.
4. 3DGS `-r 1`: images are 701 px wide (< 1600), so no automatic downscaling happens.
5. Metrics: SSIM is identical to 3DGS's implementation; LPIPS uses the official `lpips`
   package (VGG and AlexNet) on [-1,1] inputs. 3DGS's own metrics.py feeds [0,1] images to
   LPIPS, so its LPIPS number differs.
