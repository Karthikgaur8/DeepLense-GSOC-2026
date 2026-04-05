# Specific Test IX — Foundation Model for Gravitational Lensing

## Overview
Task IX.A trains a Masked Autoencoder (MAE) on unlabeled strong lensing images to learn self-supervised feature representations, then fine-tunes for multi-class classification. Task IX.B reuses the pretrained encoder for 2× super-resolution of lensing images.

---

## Task IX.A — Self-Supervised Pretraining + Classification

### Architecture
| | |
|---|---|
| Architecture | MAE (ViT-Small encoder, 8×8 patches) |
| Encoder | dim=384, depth=12, heads=6 |
| Decoder | dim=192, depth=4, heads=4 |
| Parameters | 21.3M (encoder + classification head) |

### MAE Pretraining
| | |
|---|---|
| Data | 29,449 no_sub images (unlabeled) |
| Mask ratio | 0.75 |
| Epochs | 200 |
| Objective | Reconstruct masked patches from visible patches |

### Classification (Fine-tuned)
| | |
|---|---|
| Data Split | 80,193 train / 8,911 test (90/10) |
| Classes | no_sub, cdm, axion |
| Epochs | 50 |
| Test Accuracy | 0.9899 |

### Per-Class AUC (One-vs-Rest)
| Class | AUC |
|---|---|
| no_sub | 1.0000 |
| cdm | 0.9990 |
| axion | 0.9987 |
| **Macro** | **0.9992** |

### Confusion Matrix
| Predicted → | no_sub | cdm | axion |
|---|---|---|---|
| **no_sub** | 2945 | 0 | 0 |
| **cdm** | 3 | 2916 | 57 |
| **axion** | 0 | 30 | 2960 |

---

## Task IX.B — Super-Resolution (2×)

### Architecture
| | |
|---|---|
| Encoder | Pretrained MAE ViT-Small (from Task IX.A) |
| Pos Embedding | Interpolated 8×8 → 10×10 for resolution change |
| Decoder | CNN upsampler ([256, 128, 64, 32]) |
| Parameters | 22.2M (encoder: 21.4M, decoder: 0.8M) |

### Training
| | |
|---|---|
| Data Split | 9,000 train / 1,000 test (90/10) |
| Epochs | 50 |
| Loss | 70% MSE + 30% (1 − SSIM) |
| Scale | 2× (75×75 → 150×150) |

### Test Metrics
| Metric | Value |
|---|---|
| MSE | 0.000072 |
| SSIM | 0.9732 |
| PSNR | 41.46 dB |

---

## Files
| File | Description |
|---|---|
| `Test IX A KG.ipynb` | Task IX.A notebook (MAE pretraining + classification) |
| `Test IX B .ipynb` | Task IX.B notebook (super-resolution) |
| `mae_encoder_best.pth` | Pretrained MAE encoder weights |
| `mae_full_final.pth` | Full MAE weights (used in Task IX.B) |
| `classifier_bestkg.pth` | Fine-tuned classification head |
| `sr_model_final.pth` | Fine-tuned super-resolution model |
