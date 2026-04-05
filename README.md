# DeepLense — GSoC 2026 Test Submissions

## Results Summary

### Classification Tasks
| Test | Model | Macro AUC | No Sub | Sphere/CDM | Vort/Axion | Accuracy |
|---|---|---|---|---|---|---|
| Common Test I | ResNet-18 (Pretrained) | 0.9940 | 0.9948 | 0.9916 | 0.9956 | 95.8% |
| Specific Test VII | Physics-Informed ResNet-18 : SIS lensing layer, learned θ_E | **0.9953** | 0.9958 | 0.9924 | 0.9978 | 96.2% |
| Specific Test IX.A | MAE ViT-Small | **0.9992** | 1.0000 | 0.9990 | 0.9987 | 98.9% |

### Super-Resolution (Test IX.B)
MAE encoder pretrained on no_sub (Task IX.A), fine-tuned with residual learning on bicubic + encoder features.
| Model | MSE | SSIM | PSNR |
|---|---|---|---|
| Bicubic (baseline) | 0.000070 | 0.9728 | 41.60 dB |
| **MAE-SR** | **0.000062** | **0.9761** | **42.10 dB** |

## Repository Structure
| Folder | Description |
|---|---|
| [`Common_Test/`](Common_Test/) | ResNet-18 baseline classifier |
| [`Specific_Test_VII/`](Specific_Test_VII/) | Physics-Informed ResNet-18 (PIRN - SIS lensing layer, learned θ_E) |
| [`Specific_Test_IX/`](Specific_Test_IX/) | Foundation model (MAE pretraining + classification + super-resolution) |

## Strategy
For detailed strategy discussion, architecture decisions, and training analysis, see the notebook in each test folder.
