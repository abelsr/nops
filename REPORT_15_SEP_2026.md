# Report — Navier-Stokes FNO Experiments
## Session: 15 September 2026

> Supersedes the results in [`REPORT_14_JUL_2026.md`](REPORT_14_JUL_2026.md).
> Commits: `a0f47d6` (this session).

---

## Executive Summary

This session improved the 2D forced Navier-Stokes (ν=1e-3) FNO from the
previously reported best of **val_rell2 ≈ 0.112** to:

| Metric | Before (14 Jul) | **Final (15 Sep)** | Change |
|--------|-----------------|--------------------|--------|
| **val_l2** (best) | 0.112 | **0.0509** | **−55%** |
| **test_l2** | 0.1121 | **0.0528** | **−53%** |
| test_l1 | — | 0.0350 | — |
| test_energy_err | — | 0.0032 | — |
| **Parameters** | 8,574,337 | **2,759,377** | **−68%** |
| Epochs | 50–150 | 250 | — |
| **Oracle target norm?** | n/a | **no** | — |

**A 3× smaller model, 55% more accurate — and the final number is
oracle-free.** It also **beats the earlier "SpectraNet target" of 0.0822 by
38%** (that target now looks like a mismatched reference — it matches
FNO-3D's ν=1e-4/N=10000 figure of 0.0820).

> **Read this before quoting 0.0509.** An earlier result in this session
> (0.0580) was measured with the *true target L2 norm* supplied externally —
> the model was scale-free, so the metric silently used an oracle. The final
> model (`target_scale=window`) predicts physically-scaled output and needs
> no oracle, so **0.0509 is the honest, comparable number** — and it is
> better than the oracle-assisted 0.0580.

---

## 1. What Changed

### 1.1 Temporal context (the dominant lever)

The previous setup predicted the next frame from a **single** frame. The
original FNO paper's `FNO-2D` Navier-Stokes benchmark instead maps the
**previous 10 time steps → next step**. Adding that context was the single
biggest win.

- New `MultiFrameDataset` in `experiments/navier-stokes/data/ns_loader.py`
- Each sample: `[10, 64, 64]` (K frames as channels) → `[64, 64]`

### 1.2 Native spectral convolution (`NativeSpectralConv`)

Replaces the previous real/imag split with **per-forward Tucker
reconstruction** — flagged as the bottleneck in the 14 Jul report.

The new layer is a faithful re-implementation of Li et al. (2020):

- Single `torch.cfloat` weight block per sign-quadrant (`2^(N-1)` blocks:
  1 for 1-D, 2 for 2-D, 4 for 3-D)
- `rfftn` / `irfftn` with the last axis kept at non-negative frequencies
- FFT forced to `float32` → AMP-safe
- Wired through `FourierBlock` → `FNO` via the `native_spectral_conv` flag

### 1.3 Mixed precision (AMP)

Added `torch.autocast` + `GradScaler` behind `training.amp` (default on for
CUDA). Roughly 1.5–2× epoch throughput, which is what made longer runs
tractable on the 4 GB RTX 3050.

### 1.4 Full data + longer schedule

- Training trajectories: 800 → **1000** (10,000 context samples)
- Epochs: 150 → **250**

This is what turned the *test* score around: val improved only 0.0584 → 0.0580,
but **test improved 0.0658 → 0.0604**, i.e. the extra data bought
generalisation rather than memorisation.

### 1.5 Bug fixes

| File | Fix |
|------|-----|
| `trainer.py` | removed unreachable duplicate `return` in `evaluate()` |
| `trainer.py` | `build_model` now forwards `native_spectral_conv` |
| `trainer.py` | `resolution_aware` default now matches the model |
| `trainer.py` | `spectral_gradient_penalty` view shape (`rfftfreq` length is `w//2+1`) |

---

## 2. Experiment Results

All runs: trajectory split (seed 42), per-field L2 normalisation, AdamW,
cosine schedule, AMP on, RTX 3050 Laptop (4 GB).

| # | Experiment | Params | val_l2 | test_l2 | Verdict |
|---|------------|--------|--------|---------|---------|
| 0 | Previous best (single frame, legacy conv) | 8.57M | 0.1120 | 0.1121 | baseline |
| 1 | **`fno_ctx`** — 10-frame context + native conv, 150 ep | **2.76M** | **0.0584** | 0.0658 | 🏆 breakthrough |
| 2 | `fno_ctx16` — modes 16×16, wider, 150 ep | 8.57M | 0.0591 | 0.0671 | ❌ worse |
| 3 | `fno_ctx` + gradient loss (λ=0.1), 150 ep | 2.76M | 0.0585 | 0.0660 | ❌ neutral |
| 4 | `fno_ctx` + 1000 traj + 250 ep | 2.76M | 0.0580 | 0.0604 | ✅ (oracle-assisted) |
| 5 | **`fno_ctx` + `target_scale=window`, 250 ep** | **2.76M** | **0.0509** | **0.0528** | 🏆 **best, oracle-free** |
| — | `fno3d` — space-time (FNO-3D style), 35 ep (stopped) | 16.2M | 0.2334 | — | ❌ plateaued |

### 2.1 Per-frame error (best model, test trajectory 1000)

| Target t | 14 Jul model | ctx (ep150) | **final (ep250)** |
|----------|--------------|-------------|-------------------|
| t=10 | 0.233 | 0.024 | **0.021** |
| t=13 | 0.042 | 0.027 | **0.024** |
| t=15 | 0.042 | 0.028 | **0.026** |
| t=17 | 0.068 | 0.037 | **0.033** |
| **t=19** | **0.203** | 0.053 | **0.043** |
| max err @ t=19 | 2.071 | 0.622 | **0.571** |

Figures: `outputs/predictions_ep50.png` (old),
`outputs/predictions_ctx_ep150.png`,
`outputs/predictions_final_ep250.png` (best).

---

## 3. Negative Results (load-bearing)

Two hypotheses were **falsified** — these narrow down where the remaining
gap actually lives.

### 3.1 Capacity is not the limiter

`modes=[16,16]`, `mid_channels=64` (8.57M params) scored **worse** than
`modes=[12,12]`, `mid_channels=48` (2.76M): 0.0591/0.0671 vs 0.0584/0.0658.
Tripling the parameters bought nothing. Widening the model further is
wasted compute.

### 3.2 The objective is not the limiter

A spectral gradient penalty
`L = MSE(pred, true) + λ·MSE(∇pred, ∇true)`, λ=0.1, was **neutral**:
0.0585 vs 0.0584. Plain MSE on the normalised field already fits the
derivative structure, consistent with `v_E ≈ 0.003` (energy already matched).

### 3.3 Space-time (FNO-3D) did not help

A 3-D space-time model (predict frames 10–19 from 0–9 as an `[H,W,T]`
volume) plateaued at **0.2334** by epoch 35. Note this is **not directly
comparable** to the 1-step numbers — it predicts all 10 future frames at
once, so error compounds over the whole horizon.

---

## 4. Comparison with the Original FNO Paper

Li et al. (2020), Table 1 — Navier-Stokes, 64×64:

| Model | Params | ν=1e-3 |
|-------|--------|--------|
| FNO-3D | 6,558,537 | 0.0086 |
| FNO-2D | 414,517 | 0.0128 |
| U-Net | 24,950,491 | 0.0245 |
| TF-Net | 7,451,724 | 0.0225 |
| **Ours (this session)** | **2,759,377** | **0.0580** |

We are **~4.5× above FNO-2D**. The comparison is **not clean**, for reasons
that are now well understood:

1. **Horizon / regime** — the paper's ν=1e-3 run uses T=50, N=1000; our file
   is `N1200_T20`.
2. **Metric definition** — the paper reports a trajectory-level relative
   error; we report per-sample single-step relative L2.
3. **Context** — the paper's FNO-2D baseline is a 2D+RNN model; we now match
   its 10-frame context, which is what closed most of our gap.

Because the two remaining falsified hypotheses (capacity, objective) point
away from model design, the residual gap is most plausibly **data / regime /
evaluation protocol**, not architecture.

---

## 4.1 Evaluation-protocol check (measured, 15 Sep follow-up)

The same predictions from `data/ctx_long/checkpoints/ep250.pth` were scored
under several metric conventions (`experiments/navier-stokes/eval_protocol.py`):

| Metric variant | Value |
|----------------|-------|
| 1. per-sample mean (true per-sample relL2) | **0.0470** |
| 2. per-batch ratio (FNO repo style) | 0.0616 |
| 3. global ratio (whole test set) | 0.0627 |
| 4. global ratio, physical domain | 0.0690 |
| 5. physical ratio with a *predicted* scale | **0.2930** |

### Findings

1. **Metric convention explains ~1.5× of the gap, not all of it.**
   Moving from per-sample averaging to a physical-domain global ratio moves
   the number `0.047 → 0.069` (×1.47). Against the paper's 0.0128 the gap
   therefore narrows from ~4.5× to ~3.7×, but does **not** close. Roughly
   3× remains a genuine model/regime difference.

2. **The model is scale-free — the metric so far has used an oracle.**
   Because every target is normalised by *its own* L2 norm, the model never
   learns absolute amplitude. Our per-sample metric cancels that norm, which
   is why it looks good. Replacing the oracle target norm with a scale
   predicted from the input window's RMS inflates the physical-domain error
   to **0.293** (variant 5). The mean amplitude ratio `n_tgt / n_win = 1.434`,
   i.e. the flow amplitude grows ~43% across a 10-frame window.

3. **A true autoregressive rollout is not currently possible.**
   With the predicted scale, the rollout degrades immediately (step 1 =
   0.394) and saturates to ≈0.99 by step 10 — the prediction becomes
   decorrelated from the truth. This is dominated by cumulative scale drift,
   not by a dynamics failure.

### Implication

The paper's **trajectory-level** metric cannot be reproduced with the current
training scheme, because the model does not predict absolute amplitude. A
faithful trajectory-level comparison requires **retraining so the model
outputs a physically scaled field** — either by training on raw fields (as
the original FNO does) or by normalising the target by the *input window*
scale rather than by its own norm. That retrain is now the highest-value
next step, and it should be evaluated with the `eval_protocol.py` harness
before any further architecture work.

---

## 4.2 Fixing the oracle: `target_scale=window` retrain

### The fix

Both the input window **and** the target are divided by the *window's* RMS
(instead of normalising the target by its own norm). The model therefore
outputs a physically-scaled field, and de-normalisation needs only the input
window — information genuinely available at inference time.

Selected via `training.target_scale=window` (default `"own"` keeps the legacy
behaviour). Implemented as `MultiFrameScaledDataset` +
`make_dataloaders_scaled`.

### Result — `FNO2D_scaled_250ep`

| Metric | Value |
|--------|-------|
| best val_l2 | **0.0509** (ep243) |
| test_l2 | **0.0528** |
| test_l1 | 0.0350 |
| test_energy_err | 0.0032 |
| params | 2,759,377 |
| epochs | 250 |

The scaled model **beats the oracle-assisted model on every metric variant**
and is measured honestly:

| Metric variant | `own` (oracle) | **`window` (oracle-free)** |
|----------------|----------------|-----------------------------|
| per-sample mean | 0.0470 | **0.0453** |
| per-batch ratio (FNO repo) | 0.0616 | **0.0537** |
| global ratio | 0.0627 | **0.0544** |
| global ratio, physical | 0.0690 | **0.0670** |
| scale ablation | 0.2930 (guessed scale) | n/a — model predicts scale |

### Autoregressive rollout (the real payoff)

`target_scale=own` could not be rolled out at all (error saturated at ≈0.99
by step 10 — prediction decorrelated from truth). The scaled model rolls out
cleanly:

| Step (t) | 1 (11) | 2 (12) | 5 (15) | 8 (18) | 10 (20) |
|----------|--------|--------|--------|--------|---------|
| `own` (guessed scale) | 0.394 | 0.459 | 0.652 | 0.898 | **0.993** ❌ |
| **`window` (oracle-free)** | **0.0235** | **0.0305** | **0.0547** | **0.1126** | **0.1814** ✅ |

Mean over rollout steps: **0.0782**. Error grows gracefully (×7.7 over 10
steps) instead of diverging — which is what a usable operator should do.

### Per-frame error (scaled model, test trajectory 1000)

| Target t | 14 Jul model | oracle `own` (ep250) | **scaled (ep250)** |
|----------|--------------|----------------------|--------------------|
| t=10 | 0.233 | 0.021 | **0.020** |
| t=13 | 0.042 | 0.024 | **0.022** |
| t=15 | 0.042 | 0.026 | **0.024** |
| t=17 | 0.068 | 0.033 | **0.033** |
| t=19 | 0.203 | 0.043 | 0.047 |
| **max err @ t=19** | 2.071 | 0.571 | **0.455** |

Figure: `outputs/predictions_scaled_ep250.png`.

### Interpretation

The window-scaled scheme wins for two reasons:

1. The loss is on physically-scaled targets, so high-amplitude (late, hard)
   frames receive proportionally more gradient weight instead of every frame
   being forced to unit RMS.
2. The absolute amplitude in the input window is no longer discarded — it
   carries information about where in the trajectory the flow is.

Both mattered, and neither was visible under the oracle metric.

---

## 5. Artifacts & Reproduction

### Files added / changed

| File | Change |
|------|--------|
| `nops/fno/layers/spectral_convolution.py` | `NativeSpectralConv` (new); legacy `SpectralConvolution` kept |
| `nops/fno/layers/fno_block.py` | `native_spectral_conv` flag |
| `nops/fno/models/original.py` | passes flag to blocks |
| `experiments/navier-stokes/data/ns_loader.py` | `MultiFrameDataset`, `MultiFrameScaledDataset`, `SpaceTimeDataset`, `make_dataloaders_{ctx,scaled,3d}` |
| `experiments/navier-stokes/trainer.py` | AMP, gradient loss, generalised forward, `target_scale`, fixes |
| `experiments/navier-stokes/plot_predictions.py` | prediction-vs-truth figures |
| `experiments/navier-stokes/eval_protocol.py` | metric-definition + rollout harness |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | containerised workflow |
| `configs/v1/model/{fno_ctx,fno_ctx16,fno3d}.yaml` | model configs |

### Checkpoints

- **Best (use this)**: `data/ctx_scaled/checkpoints/ep250.pth`
  — val_l2 = **0.0509**, test_l2 = **0.0528**, `target_scale=window`, oracle-free
- `data/ctx_long/checkpoints/ep250.pth` — val_l2 = 0.0580 (oracle-assisted)
- `data/ctx_grad/checkpoints/ep150.pth` (gradient-loss run)
- ⚠️ The single-frame-era `data/checkpoints/ep50.pth` (0.1127) was
  **overwritten** by a later run before per-experiment checkpoint dirs were
  introduced. Metrics and plots survive; weights do not.

### Reproduce the best result

```bash
docker compose run --rm nops uv run python \
  experiments/navier-stokes/trainer.py \
  model=fno_ctx \
  training.target_scale=window \
  training.epochs=250 \
  training.train_samples=1000 training.val_samples=100 training.test_samples=100 \
  training.batch_size=16 training.val_batch_size=50 \
  training.weight_decay=0.001 training.checkpoint_interval=50 \
  training.data_dir=./data/ctx_scaled \
  exp_name=FNO2D_scaled_250ep
```

Plot the result:

```bash
docker compose run --rm nops uv run python \
  experiments/navier-stokes/plot_predictions.py \
  --ckpt data/ctx_scaled/checkpoints/ep250.pth \
  --out outputs/predictions_scaled_ep250.png
```

Score it (metric variants + oracle-free rollout):

```bash
docker compose run --rm nops uv run python \
  experiments/navier-stokes/eval_protocol.py \
  --ckpt data/ctx_scaled/checkpoints/ep250.pth
```

### Note on checkpoints

Use a per-experiment `training.data_dir` (e.g. `./data/ctx_long`) to keep
checkpoints isolated — otherwise runs at the same epoch count overwrite
each other.

---

## 6. Next Steps (priority order)

1. **Longer schedule** — the scaled model was still improving at epoch 250.
   The paper trains 500 epochs; a 500-epoch scaled run is the cheapest
   remaining gain.
2. **Old evaluation-protocol caveats are now resolved** — the model is
   oracle-free and rollable, so future results are directly comparable.
   Quote the **rollout mean (0.0782)** alongside the single-step number when
   comparing to the paper's trajectory-level figures.
3. **Super-resolution** — FNO's headline property. The `resolution_aware`
   (MFI) path exists but is currently disabled; a cross-resolution test
   would be a stronger demonstration than further absolute-error tuning.
4. **Ablate against raw fields** — `target_scale=window` beat `own`; it is
   worth testing plain raw-field training (what the original FNO does) on
   the same harness to see whether the residual normalisation helps or hurts.

*Report generated: 15 September 2026*
