# Specific Test IX.A — Masked Autoencoder for Gravitational Lensing Classification

## Model
| | |
|---|---|
| Architecture | MAE (ViT-Small encoder, 8x8 patches) |
| Encoder | dim=384, depth=12, heads=6 |
| Decoder | dim=192, depth=4, heads=4 |
| Parameters | 21.3M (encoder+head) |

## Pretraining
| | |
|---|---|
| Data | 29,449 no_sub images |
| Mask ratio | 0.75 |
| Epochs | 200 |

## Classification
| | |
|---|---|
| Split | 80,193 train / 8,911 test (90/10) |
| Epochs | 50 |
| Test Accuracy | 0.9899 |

## Per-Class AUC (One-vs-Rest)
| Class | AUC |
|---|---|
| no_sub | 1.0000 |
| cdm | 0.9990 |
| axion | 0.9987 |
| **Macro** | **0.9992** |

## Confusion Matrix
| | no_sub | cdm | axion |
|---|---|---|---|
| no_sub | 2945 | 0 | 0 |
| cdm | 3 | 2916 | 57 |
| axion | 0 | 30 | 2960 |

## Saved Weights
- ``mae_encoder_best.pth`` — pretrained encoder
- ``mae_full_final.pth`` — full MAE (for Task IX.B)
- ``classifier_bestkg.pth`` — fine-tuned classifier
