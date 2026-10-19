# Scripts: InstantNGP vs 3DGS

All commands run from the assignment root (`assignment2/`).

## Laptop (done)
| Script | What it did |
|---|---|
| `filter_frames.py` | sharpest frame per group -> `data/scene/input/` |
| `prepare_ngp.py` | recentre `transforms.json`, every-8th split -> `transforms_{train,test}.json`, `results/{train,test}_list.txt` |
| `make_novel_paths.py` | 7 novel camera paths -> `data/scene/novel/*.json` (both 3DGS and InstantNGP poses per frame, verified to project identically) |

## Server
Copy to the server (same folder layout):
```
data/scene/images  data/scene/sparse  data/scene/colmap_text  data/scene/novel
data/scene/transforms_train.json  data/scene/transforms_test.json
results/test_list.txt  results/train_list.txt  scripts/
```
Clone both repos into `repos/`, build the `gs` and `ngp` conda envs, then in the `gs` env:
`pip install lpips matplotlib scipy`.

```
tmux new -s a2
DRY_RUN=1 bash scripts/run_all.sh          # prints every command, runs nothing
bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
```
or step by step: `run_gs.sh`, `run_ngp.sh`, `run_eval.sh`. Env names / iteration lists: `config.sh`.

| Script | Env | Does |
|---|---|---|
| `run_gs.sh` | gs | train 30k (saves 1k/3k/7k/15k/30k, timed, GPU memory logged), render test views per checkpoint, benchmark, novel paths, official render.py+metrics.py cross-check |
| `run_ngp.sh` | ngp | `train_ngp.py` 35k (snapshots 1k/3k/7k/15k/35k, timed), `render_ngp.py` test views per checkpoint, benchmark (spp 1 and 8), novel paths |
| `run_eval.sh` | gs | `eval_metrics.py` for every checkpoint, `collect_results.py`, `make_figures.py` |

Outputs: `results/summary.md` (tables), `results/summary.json`, `results/figures/*.png`, `results/videos/*.mp4`.

## Fairness built in
- same undistorted images (701x1249), same COLMAP poses, same 54/8 split
- one metric implementation for both (SSIM identical to 3DGS's; LPIPS = official package)
- timing: warm-up + CUDA sync (3DGS) / host copy (NGP), same GPU, GPU-idle check logged
- novel views: identical cameras for both methods (checked at generation and again in render_gs.py)
