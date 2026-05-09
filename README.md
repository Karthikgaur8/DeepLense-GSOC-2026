# DeepLense — GSoC 2026 Test Submissions

**Foundation Model for Gravitational Lensing.** Solutions to the Common Test (multi-class classification), Specific Test VII (Physics-Guided ML), and Specific Test IX.A / IX.B (Foundation Model: classification + super-resolution) under the DeepLense project — Alabama, MIT, PSL, IISc.

The four notebooks form a deliberate progression: a strong supervised baseline → a physics-informed network with a learnable Einstein radius → a self-supervised foundation model whose encoder is reused, unchanged, across two different downstream tasks. Each notebook is self-contained and reproducible end-to-end on a single GPU.

This repository is my GSoC 2026 contributor test submission for ML4SCI's NSF-funded DeepLense project. I'm continuing with the project as a mentor for the 2026 cycle.

---

## Results

### Classification (Macro AUC, one-vs-rest)

| Test | Model | Macro AUC | Per-class (No-sub / CDM-sphere / Axion-vortex) | Accuracy |
|------|-------|-----------|------------------------------------------------|----------|
| Common Test I | ResNet-18, ImageNet-pretrained | **0.9940** | 0.9948 / 0.9916 / 0.9956 | 95.8% |
| Specific VII (PINN) | Physics-Informed ResNet-18, learnable θ_E | **0.9953** | 0.9958 / 0.9924 / 0.9978 | 96.2% |
| Specific IX.A | MAE ViT-Small, self-supervised pretrain → fine-tune | **0.9992** | 1.0000 / 0.9990 / 0.9987 | 98.9% |

### Super-Resolution (Specific IX.B, 75×75 → 150×150)

| Model | MSE | SSIM | PSNR |
|-------|------|------|------|
| Bicubic baseline | 0.000070 | 0.9728 | 41.60 dB |
| **MAE-SR (this work)** | **0.000062** | **0.9761** | **42.10 dB** |

The MAE encoder pretrained for IX.A is reused, unchanged, as the backbone for IX.B. Cross-task transfer is the headline result, not the per-task numbers.

---

## What the tests are, and why they matter

Strong gravitational lensing is one of the cleanest empirical probes of dark-matter substructure. The morphology of an Einstein ring carries imprints of the lens's mass distribution, and small perturbations in the ring distinguish dark-matter models — Cold Dark Matter (CDM) subhalos, axion-like particle vortex defects, or no substructure. Robust ML for these images is an active research area at MIT and Alabama (Alexander et al. 2020; Reddy et al. 2024); progress here directly affects which dark-matter hypotheses are testable with upcoming surveys (Euclid, LSST, Roman).

The three tests cover the three approaches a serious lensing-ML practitioner is expected to know:

1. **Common Test I** — supervised baseline. Establishes that a clean CV pipeline already crosses AUC ≈ 0.99 on this dataset.
2. **Specific Test VII** — physics-guided ML. The lensing equation goes inside the network. The hypothesis: a domain prior beats a pure CV baseline.
3. **Specific Test IX (A + B)** — foundation model. Pretrain a backbone label-free on lensing images (IX.A), then fine-tune for two unrelated downstream tasks: classification and super-resolution. This is the hardest test because it forces the same encoder to generalize across tasks.

---

## Common Test I — Multi-Class Classification

**Task.** Three-way classification on 150×150 single-channel simulated lensing images (no-sub / spherical CDM subhalo / vortex string defect). Evaluation: ROC + AUC.

**Approach.** ResNet-18 with ImageNet weights, adapted for 1-channel input by averaging the pretrained RGB `conv1` weights along the channel axis. This preserves the learned edge/curve filters rather than starting from random init. Final FC reset for 3-way output. ~11.2M parameters — well-matched to 33,750 training samples without excessive overfitting risk.

**Engineering decisions.**
- **Augmentation: random h/v flips and 180° rotations** — gravitational lensing images have no preferred sky orientation, so all rotations and reflections are physically valid.
- **Stratified 90:10 split** of the combined provided data, per submission guidelines.
- **Adam (lr=1e-3, wd=1e-4) + CosineAnnealingLR over 30 epochs.** Best checkpoint by validation loss.
- Training time: ~3.5 min on a single H200.

**Result.** Macro AUC 0.9940. Consistent with reported AUCs in the literature for this architecture-dataset pair (Alexander et al. 2020) — a sanity check on the baseline before adding complexity. The subhalo class is the hardest (AUC 0.9916), consistent with the underlying physics: spherical subhalo perturbations are localized and subtle compared to vortex string defects.

---

## Specific Test VII — Physics-Guided ML (PINN)

**Task.** Same classification problem, but the architecture must be a Physics-Informed Neural Network using the gravitational lensing equation. Beat the Common Test result.

**Why this is non-trivial.** Most "physics-informed" image ML in practice means computing physics-motivated features in the dataloader and passing them as extra channels — the network never sees the equation. A real PINN here means the lensing equation runs *inside* `nn.Module.forward()` and gradients flow through it.

**Approach.** A custom `LensPINN` module:

- The Singular Isothermal Sphere lens equation
  ```
  β = θ − θ_E · θ/|θ|
  ```
  is implemented as a coordinate remap inside `forward()` via `F.grid_sample`. The Einstein radius `θ_E` is an `nn.Parameter` (parameterized as `log θ_E` for positivity), so the classification loss backpropagates *into the physics parameter itself*.
- Three input channels are stacked for a ResNet-18 backbone:
  1. Raw lensing image
  2. **Azimuthal residual** — pixel intensity minus the mean intensity at that radius. This isolates the deviation from radial symmetry, which is precisely the substructure signal.
  3. **SIS lens remap** with the learned θ_E — the input as the network "sees" the source plane.

**Training (two-phase, to handle large `grid_sample` gradients):**
- **Phase 1 (25 epochs):** θ_E frozen at 0.32, backbone trains with Adam (lr=1e-3) + cosine schedule.
- **Phase 2 (10 epochs):** θ_E unfrozen at lr=1e-4 for fine-tuning. The learned θ_E shifts from 0.3200 → 0.3206 — small in magnitude, enough to push macro AUC from 0.9938 to 0.9953.

**Validation before integration.** Before assembling the full PINN, I built standalone `ThetaEncoder` and `LensInversion` modules and visually confirmed that at the correct θ_E ≈ 0.327 the Einstein ring collapses toward a compact source galaxy. That ruled out grid-sample sign errors and confirmed gradients flow through the inversion as expected.

**Result.** Macro AUC 0.9953 — a +0.13% improvement over the Common Test baseline. The improvement is modest, which is itself the honest finding: at this dataset size and AUC level, the physics prior offers a small but real margin over a strong CV baseline. The vortex-class AUC sees the largest gain (0.9956 → 0.9978), consistent with the azimuthal-residual channel directly capturing the broken radial symmetry that defines the vortex class.

Preprocessing inspired by LensPINN (Ojha et al.) is implemented and visualized in the notebook (Section 3), but the final model uses the azimuthal-residual channel, which empirically performed better on this dataset.

---

## Specific Test IX.A — MAE Pretraining + Classification

**Task.** Train a Masked Autoencoder on `no_sub` images only, then fine-tune for 3-class classification (no_sub / cdm / axion). Different dataset and a different substructure pair from the Common Test.

**Why MAE.** The discriminative signal between substructure classes lives in subtle spatial features — ring sharpness, localized clumps, broken azimuthal symmetry — that a pixel-reconstruction objective directly forces the encoder to capture. JEPA-style objectives are an alternative, but reconstruction is more sample-efficient at this dataset size (~30k pretraining images).

**Preprocessing — driven by data analysis, not defaults.** Pixel values span eight orders of magnitude (1e-10 to ~4.58) with extreme right skew (median ≈ 2,800× smaller than mean). Raw or min-max normalization wastes encoder capacity on a few rare bright pixels. I apply `log1p` followed by per-image z-score, which compresses dynamic range and spreads the near-zero structure where the physics actually lives.

**Architecture.**
- ViT-Small encoder: dim=384, depth=12, 6 heads.
- **8×8 patches** → 64 tokens per 64×64 image. Patch size is set so each patch is smaller than the Einstein-ring width (peaks at radius ~8 px); finer patches preserve the structure that matters.
- Lightweight decoder (dim=192, depth=4), since it's discarded after pretraining.
- **75% mask ratio** — justified by the steep power spectrum of these images: smooth, low-frequency content tolerates aggressive masking.
- Model scale follows benchmarks from the same group's Lens-JEPA work on comparable dataset sizes.

**Pretraining.** AdamW (lr=1.5e-4, betas=(0.9, 0.95), wd=0.05), cosine schedule with 20-epoch warmup. Augmentations limited to physics-justified transforms (90° rotations, h/v flips).

**Fine-tuning.** Differential learning rates: pretrained encoder at 1e-5, fresh classification head at 1e-4. Label smoothing 0.1. The differential-LR practice matches what the lab uses in their published work — preserves pretrained features while letting the head adapt.

**Result.** Macro AUC **0.9992**, accuracy 98.9%. Per-class: no_sub 1.0000, cdm 0.9990, axion 0.9987. The encoder weights (`mae_encoder_best.pth`) are saved and reused directly in IX.B.

---

## Specific Test IX.B — Super-Resolution from the Same MAE Encoder

**Task.** 2× super-resolution on lensing images (75×75 → 150×150). Evaluation: MSE, SSIM, PSNR.

**The point of this test.** It's not "build a super-resolution model." It's "show that one pretrained backbone serves more than one task." That's what makes this a foundation-model test rather than a regression test.

**Resolution adaptation.** The MAE was pretrained on 64×64 (8×8 patch grid, 64 tokens). LR inputs are 75×75, so I pad to 80×80 with reflect padding (10×10 grid, 100 tokens). Positional embeddings are bicubically interpolated from 8×8 → 10×10, preserving learned spatial relationships. Transformer weights transfer directly since self-attention is resolution-agnostic.

**Architecture — residual learning, not pixel-from-scratch.** A naïve approach generates HR pixels directly from abstract encoder tokens. In practice this collapses to roughly bicubic-quality output because there's no inductive bias toward image fidelity. Following standard SR literature (VDSR, EDSR, SwinIR), I use **residual learning on bicubic + encoder features**:

1. Compute bicubic upsampling as a free baseline.
2. Project and bilinearly upsample the encoder's spatial features to 150×150.
3. Concatenate with the bicubic image and pass through a 6-layer CNN refinement network operating at full HR resolution.

The CNN learns *targeted corrections* — sharpening ring edges, recovering source structure, restoring detail bicubic smooths over. This guarantees the output is at least as good as bicubic; any learning only improves from there.

**Loss.** 70% MSE + 30% (1 − SSIM). MSE for pixel accuracy, SSIM for the structural features that matter for downstream lensing science.

**Training.** Differential LRs again — encoder at 1e-5, refinement at 3e-4. AdamW, cosine schedule with 5-epoch warmup, 100 epochs total.

**Result.** MSE 0.000062 (vs 0.000070 bicubic, **11% reduction**), SSIM 0.9761 (vs 0.9728), PSNR 42.10 dB (+0.50 dB). All three metrics improve over bicubic on the held-out test set (1,000 images). The lab's own DiffLense work (Reddy et al. 2024) reports PSNR in a comparable range on their LR/HR pipelines.

**The cross-task transfer is the actual deliverable.** The same `mae_encoder_best.pth` checkpoint produces a 0.9992-AUC classifier *and* a super-resolution model that beats bicubic on every metric. That's the foundation-model claim made operational.

---

## Repository Structure

```
.
├── Common_Test/
│   └── Common_Test.ipynb           # ResNet-18 baseline (0.9940 macro AUC)
├── Specific_Test_VII/
│   └── Specific_Test_PINN.ipynb    # Physics-Informed ResNet-18, learnable θ_E (0.9953)
└── Specific_Test_IX/
    ├── Test_IX_A.ipynb             # MAE pretrain → classification (0.9992)
    └── Test_IX_B.ipynb             # MAE encoder → 2× super-resolution
```

Each notebook is self-contained: data loading, model definition, training, evaluation, and visualizations. A short strategy discussion sits at the top of every notebook, explaining design choices and what was tried before the final approach.

---

## Reproducing

- **Hardware.** Single NVIDIA H200 (or any GPU ≥ 24 GB).
- **Stack.** PyTorch 2.x, torchvision, scikit-learn, matplotlib, numpy.
- **Datasets.** Common/PINN dataset and Foundation Model dataset are linked in the GSoC test descriptions; place under `dataset/` and `alpha/Dataset/` + `beta/Dataset/` respectively (paths configurable in each notebook).
- Run notebooks top-to-bottom. The pretrained MAE encoder from IX.A (`mae_encoder_best.pth`) is loaded by IX.B.
- Total compute across all four notebooks: ~2 GPU-hours.

---

## References

- Alexander et al. 2020 — *Deep Learning the Morphology of Dark Matter Substructure* (architectural baseline for this dataset).
- Reddy et al. 2024 — *Mach. Learn.: Sci. Technol.* 5, 035076 — *DiffLense* (super-resolution benchmark from the same group).
- Ojha et al. — *LensPINN* (physics preprocessing, explored in the PINN notebook).
- He et al. 2022 — *Masked Autoencoders Are Scalable Vision Learners*.
- Kim et al. 2016 (VDSR), Lim et al. 2017 (EDSR), Liang et al. 2021 (SwinIR) — residual super-resolution.

---

The three tests, taken together, demonstrate the foundation-model paradigm on lensing data: a single encoder, pretrained without labels, that adapts to fundamentally different downstream tasks. The numbers above are that claim, made operational.
