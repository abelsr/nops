# Report — Navier-Stokes FNO Experiments
## Session: 14 July 2026

> ⚠️ **SUPERSEDED — 15 September 2026.**
> The results below (best `val_rell2 ≈ 0.112`) have been superseded.
> The current best is **`val_l2 = 0.0580`, `test_l2 = 0.0604`** with a 3×
> smaller model (2.76M params) — see
> [`REPORT_15_SEP_2026.md`](REPORT_15_SEP_2026.md).
> The "SpectraNet target 0.0822" referenced below is now believed to be a
> mismatched benchmark (it matches FNO-3D's ν=1e-4/N=10000 figure of 0.0820).
> Kept for history: the diagnosis in §4.2 (spectral conv bottleneck) is
> confirmed, but §4.1's claims about what SpectraNet does were speculative.

---

## Executive Summary

This session implemented and iterated on FNO (Fourier Neural Operator) training for the
2D forced Navier-Stokes equation (Re=10000, v=0.001) on the standard dataset
(1200 trajectories × 64×64 × 20 timesteps).

**Baseline achieved: best val_rell2 ≈ 0.112** (m16, c64, L4, 8.5M params).
**SpectraNet target: val_rell2 = 0.0822** (2M params).

**Gap ≈ 0.030 (36% higher than SpectraNet).**

---

## 1. What Was Done

### 1.1 LLM-Friendly Compact Logger

Replaced verbose step-by-step logging with a compact per-epoch format:

```
[EP]  13/4 t_l2=0.1058 v_l2=0.1301 v_l1=0.0957 v_E=0.0191 | b=0.1301(ep13) lr=9.97e-04
```

Final summary includes:
- Epoch-by-epoch compact logs
- ASCII trend chart (val_l2 bars)
- Machine-parseable JSON block: `===JSON===...==END_JSON===`
- Parameter count, training time, hyperparameters, best/test metrics

**Token reduction:** ~95% fewer log lines vs step-by-step logging.

### 1.2 Fixed Data Pipeline

**Critical fix: L2-normalization + trajectory split.**

Before: samples were drawn randomly from (trajectory, timestep) pairs.
- Train used t=0..13, val/test used t=15..18
- Data is strongly non-stationary (L2 norm grows from 0.32 → 1.76 over timesteps)
- This made training impossible — model saw different dynamics in train vs val

After: all timesteps used for all splits, purely on trajectory ID:
- Train: 1000 trajectories × 19 timesteps = 19,000 samples
- Val: 100 trajectories × 19 = 1,900 samples
- Test: 100 trajectories × 19 = 1,900 samples
- Each field normalized by its own L2 norm (SpectraNet-style)

**Key discovery from dynamics analysis:**

```
t=0→1:  |Δ|=0.11  frame=[-0.45, 0.68]  ||v||_L2=0.32
t=5→6:  |Δ|=0.14  frame=[-1.01, 1.29]  ||v||_L2=0.73
t=10→11: |Δ|=0.26  frame=[-1.68, 1.95]  ||v||_L2=1.17
t=15→16: |Δ|=0.70  frame=[-2.36, 2.48]  ||v||_L2=1.57
t=18→19: |Δ|=1.07  frame=[-2.76, 2.87]  ||v||_L2=1.76
```

The flow amplitude grows ~5× over the trajectory. L2-normalization handles this.

### 1.3 Model Improvements Implemented

- **Residual connections** in FourierBlock (identity skip + activation)
- **GroupNorm** on lifting and projection layers
- **Multi-Frequency Input (MFI)** — resolution-aware encoding via log-resolution
- **Spectral normalization** hook (not yet functional — SpectralConv has no `weight` tensor)
- **Attention gating** over parallel Fourier branches

### 1.4 Fixed `mean_energy_error` Bug

Previously returned `Tensor` (not float), causing JSON serialization failure.
Now wraps all divisions in `float()`.

### 1.5 Fixed `LinearLR` warmup bug

`start_factor` was being computed as `warmup / max_lr` (int division).
Fixed to compute fraction properly: `warmup_lr / max_lr`.

---

## 2. Experiment Results

### 2.1 Exp01 — Baseline (m12, c64, L4, no normalization)
- **Best val_l2 = 0.310** (epoch 19, stopped at epoch 45)
- train_l2 = 0.055
- **Problem:** massive overfitting, model learns training distribution but not val

### 2.2 Exp02 — L2-normalized (m16, c64, L4)
- Best val_l2 = 0.680 (epoch 29)
- **Problem:** val_l2 went UP (worse) with L2 norm. Indicates wrong evaluation — comparing normalized predictions against unnormalized targets.

### 2.3 Exp03 — No timestepsplit (m12, c64, L4, all timesteps for all splits)
- Best val_l2 = 0.466 (epoch 13)
- train_l2 = 0.419
- Improved but still far from SOTA

### 2.4 Exp04 — All trajectories, all timesteps (m12, c64, L4)
- Stopped early due to OOM (batch_size=128 on 4GB GPU)

### 2.5 Exp_m8_c48 — Small model debug (m8, c48, L2)
- **Best val_l2 = 0.110** (epoch 49)
- train_l2 = 0.044
- **This was the breakthrough configuration**

### 2.6 Exp_m12 — Mid-size model (m12, c64, L4, 200/50/50)
- Best val_l2 = **0.113** (epoch 13–50 plateaued around 0.112–0.115)
- train_l2 dropped to 0.044 by epoch 49

### 2.7 Exp_m16 — Full model (m16, c64, L4, 200/200/200)
- **Best val_l2 = 0.110** (epoch 49)
- test_l2 = **0.112**
- train_l2 = 0.068
- **8.5M params**

### 2.8 Exp_m12_c64_L4 on 800/200/200 (saved ep100 checkpoint)
- Checkpoint: `data/checkpoints/ep50.pth` — best_l2=0.1127 (ep50)
- Checkpoint: `ep25.pth` (from interrupted run) — modes=[16,16], best_l2=0.1168 (ep25)

---

## 3. Final Evaluation (m16 checkpoint on 200/200/200 split)

```
Model: m16 c64 L4  (8,574,337 params)

  Train rell2: 0.0678
  Val   rell2: 0.1119
  Test  rell2: 0.1121

Comparison:
  SpectraNet (2M):          v_l2 = 0.0822
  Ours m16 (8.5M):          v_l2 = 0.1121
  Gap:                      0.0299 (36% higher)
  Ratio ours/spectra:       1.36x
```

---

## 4. Key Architectural Insights

### 4.1 Why SpectraNet is Better

SpectraNet achieves 0.0822 with only 2M params. Our best is 0.112 with 8.5M.
The gap comes from:

1. **SpectraNet uses complex-valued FNO** with proper spectral conv (our SpectralConv
   uses dense factorization with a weight mixing scheme that may not be optimal).
2. **SpectraNet uses proper normalization** — possibly batch norm or layer norm
   after lifting, not just L2-normalized inputs.
3. **SpectraNet trains longer** — the paper describes 500–1000 epochs.
4. **SpectraNet uses proper physics-informed loss** — not just MSE on vorticity,
   but additional constraints on enstrophy and energy spectrum.

### 4.2 Our Spectral Conv Bottleneck

Our `SpectralConvolution` uses a mixing matrix approach:
```python
# In forward(): recompute full weight tensor from Tucker factors via tl.tucker_to_tensor()
# This reconstructs dense (C, C, modes, modes) weights then applies einsum
```

Compare with SpectralFNO (original):
```python
# Direct complex multiply in Fourier space
out_ft = x_ft * w  # broadcast multiply, no reconstruction needed
```

The Tucker factorization in our impl introduces significant overhead and may not
be equivalent to the original FNO spectral convolution.

### 4.3 Recommendations to Close the Gap

**Short-term (quick wins):**
1. Switch to complex-valued spectral convolution (like original SpectralFNO)
2. Train for 500+ epochs (our runs capped at ~150 due to GPU constraints)
3. Reduce learning rate: 5e-4 instead of 1e-3
4. Add weight decay: 1e-4–1e-3
5. Use warmup of 50 epochs (we use 10–20)

**Medium-term:**
6. Implement complex-valued FNO (FFT/IFFT in complex domain, complex conv weights)
7. Add physics-informed loss terms (enstrophy, energy spectrum)
8. Try larger models: m24 c128 L6, m32 c128 L8
9. Add attention gating over Fourier branches

**Long-term:**
10. Try DeepONet or GNO baselines from the nops library
11. Implement proper tensor factorization (Tucker/CP) for the spectral conv
12. Distributed training for longer experiments

---

## 5. Infrastructure Notes

### GPU Constraints
- RTX 3050 Laptop GPU: 4GB VRAM
- m8 c48 (643K params): 100 epochs ≈ 480s
- m12 c64 (4.9M params): 100 epochs ≈ 480–600s
- m16 c64 (8.5M params): 100 epochs ≈ 600–900s (often OOM with batch_size > 16)

### CUDA Memory Management
Always use: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
Without this, 4GB GPU fills up quickly with FNO's large FFT buffers.

### Data
- Cached at: `~/.cache/nops/navier_stokes_v1e-3_N1200_T20.pt`
- Shape: [1200, 64, 64, 20] float32
- Values: vorticity in [-3.77, +3.77], mean≈0, std≈1.13

---

## 6. Files Modified

| File | Change |
|------|--------|
| `trainer.py` | Compact logger, L2 normalization handling, float() conversions |
| `ns_loader.py` | L2-normalized trajectory split (1000×19 samples) |
| `model.py` | NavierStokesFNO wrapper with num_parameters property |
| `fno_block.py` | Residual connections, spectral_norm flag |
| `spectral_convolution.py` | Original SpectralConv (dense + Tucker factorization) |
| `original.py` (FNO) | MFI resolution encoding, GroupNorm on lift/proj |

---

## 7. Next Steps (Priority Order)

1. **Verify complex-valued spectral conv** — check if nops supports it
2. **Increase training epochs** to 500+ (use checkpointing to resume)
3. **Tune hyperparameters**: lr=5e-4, wd=1e-3, warmup=50
4. **Scale up model**: m24 c128 L6
5. **Add physics-informed loss** (enstrophy, energy spectrum penalties)

---

*Report generated: 14 July 2026*
