# Results summary

## Test-view quality (mean over test views)

| Metric | InstantNGP | 3DGS |
|---|---|---|
| PSNR (dB) ↑ | 33.05 | 36.03 |
| SSIM ↑ | 0.9598 | 0.9789 |
| LPIPS-VGG ↓ | 0.1618 | 0.1469 |
| LPIPS-Alex ↓ | 0.0595 | 0.0376 |

## Efficiency

| | InstantNGP | 3DGS |
|---|---|---|
| Training iterations | 100000 | 30000 |
| Training time, wall clock (s) | 833.7 | 2593.6 |
| Render time (ms / frame) | 20.32 (spp 1) | 17.36 |
| Render speed (FPS) | 49.2 | 57.6 |
| Model size on disk (MB) | 23.35 (.ingp, compressed) | 271.75 (.ply) |
| Model size uncompressed (MB) | 36.58 | 271.75 |
| Parameters | 12,865,536 | 67,790,233 (1,148,987 Gaussians × 59) |
| Peak GPU memory, process (MiB) | – | – |
| Peak GPU memory, whole GPU (MiB) | 1209 | 4305 |

GPU: NVIDIA GeForce RTX 4060 Laptop GPU

Cross-check with the official 3DGS metrics.py on the same renders: PSNR 36.030, SSIM 0.9789 (ours 36.030, 0.9789). Its LPIPS differs because metrics.py feeds [0,1] images to a network expecting [-1,1].
