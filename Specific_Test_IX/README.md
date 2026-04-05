# Specific Test IX — Foundation Model for Gravitational Lensing

- Task IX.A: MAE self-supervised pretraining on no_sub → fine-tune for 3-class classification.  
- Task IX.B: Reuse pretrained encoder → fine-tune for 2× super-resolution.

---

## Task IX.A — Classification

**Model:** MAE ViT-Small (dim=384, 12 layers, 6 heads, 8×8 patches, 75% masking)  
**Pretraining:** 29,449 no_sub images, 200 epochs, reconstruction objective  
**Fine-tuning:** 80,193 train / 8,911 test (90/10), 50 epochs

| Class | AUC | Precision | Recall |
|---|---|---|---|
| no_sub | 1.0000 | 0.9990 | 1.0000 |
| cdm | 0.9990 | 0.9898 | 0.9798 |
| axion | 0.9987 | 0.9811 | 0.9900 |
| **Macro** | **0.9992** | | **Accuracy: 98.99%** |

## Task IX.B — Super-Resolution (2×)

**Model:** Pretrained MAE encoder (pos embed interpolated 8×8 → 10×10) + CNN residual refinement on bicubic + encoder features  
**Training:** 9,000 train / 1,000 test (90/10), 100 epochs, 70% MSE + 30% (1−SSIM)

| Model | MSE | SSIM | PSNR |
|---|---|---|---|
| Bicubic (baseline) | 0.000070 | 0.9728 | 41.60 dB |
| **MAE-SR** | **0.000062** | **0.9761** | **42.10 dB** |

---

## Files

| File | Description |
|---|---|
| `Test IX A KG.ipynb` | MAE pretraining + classification |
| `Test IX B KG.ipynb` | Super-resolution |
| `mae_encoder_best.pth` | Pretrained encoder (used by both tasks) |
| `mae_full_final.pth` | Full MAE (encoder + decoder) |
| `classifier_bestkg.pth` | Fine-tuned classifier |
| `sr_model_finalkg.pth` | Fine-tuned SR model |
