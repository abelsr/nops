# nops - autonops

This is an autonomous research experiment. You are an AI agent running experiments to find the best Neural Operator architecture and training configuration for solving the 2D Navier-Stokes equations.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). The branch `experiments/autonops/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b experiments/autonops/<tag>` from current master/main.
3. **Read the in-scope files**: Read these files completely before doing anything else:
   * `prepare.py` — fixed constants, data loading, and the canonical evaluation function. **Do not modify.**
   * `train.py` — the file you modify. Model architecture, optimizer, scheduler, loss, batch size, everything.
4. **Verify the dataset exists**: Check that the file at `DATA_PATH` (defined in `prepare.py`) exists. If not, tell the human to download it and point `NS_DATA_PATH` to it.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline is recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then immediately kick off the first experiment.

## Domain context

You are working with **Fourier Neural Operators (FNO)** applied to the **2D Navier-Stokes equations** in vorticity form:

$$\partial_t \omega + u \cdot \nabla \omega = \nu \Delta \omega + f$$

The model learns an operator mapping an initial vorticity field (T_IN timesteps) to a future vorticity field (T_OUT timesteps). This is a **spatiotemporal prediction problem** on a periodic 2D domain.

### Key concepts you must understand before experimenting

**Fourier modes (`MODES`)**: The FNO truncates the Fourier spectrum to the `k` lowest-frequency modes per dimension. Increasing modes captures finer spatial structure but increases parameter count quadratically. For 64×64 resolution, modes beyond 21 are redundant (Nyquist). A good starting range is 8–20 per spatial dimension.

**Width (`MID_CH`)**: The number of channels in the Fourier layers. This is the primary knob for model capacity. Doubling width roughly quadruples parameters. Common range: 32–256.

**Depth (`N_LAYERS`)**: Number of Fourier (or MoE) layers. More layers = more expressivity but diminishing returns after ~6 for this problem. Deeper models are also harder to optimize.

**MoE (Mixture of Experts)**: The `MoEFNO` replaces each Fourier layer with a set of expert sub-networks, gated by a learned router. `TOP_K` controls how many experts activate per sample. This can improve expressivity without proportionally increasing compute, but adds optimization complexity. If MoE is consistently underperforming plain FNO, switch `USE_MOE = False`.

**`add_grid`**: Appends normalized (x, y, t) coordinate channels to the input. Almost always helps for spatially non-uniform solutions. Keep it `True` unless you have a good reason.

**Relative L2 loss**: The metric and the loss are both relative L2 error. This is scale-invariant and the standard in the FNO literature (Li et al. 2021). Do not change the metric. You may experiment with adding auxiliary terms (e.g., H1 Sobolev loss that penalizes gradient errors), but `evaluate_rel_l2` in `prepare.py` is the ground truth.

**Baseline reference**: The original FNO paper reports `val_rel_l2 ≈ 0.1770` for ν=1e-3, T_in=10, T_out=10, 64×64. Beat this.

### Things worth trying (roughly ordered by expected impact)

1. Increase `MID_CH` (64 → 128 → 256) — often the single best lever
2. Tune `MODES` — try `[12,12,8]`, `[16,16,12]`, asymmetric modes for spatial vs time
3. Switch between `USE_MOE=True` and `USE_MOE=False` — MoE is not always better
4. Change optimizer: try `Adam` instead of `AdamW`, or add `foreach=True` for speed
5. Add H1 Sobolev loss: penalize gradient errors in addition to pointwise L2
6. Tune `LR` and `WARMUP_STEPS` — FNO is sensitive to learning rate
7. Change activation: `GELU` → `SiLU` → `Tanh` (Tanh sometimes helps for smooth PDE solutions)
8. Add layer normalization inside the Fourier blocks (modify `build_model`)
9. Increase `N_LAYERS` (4 → 6) with reduced `MID_CH` to keep params similar
10. Try a OneCycleLR scheduler instead of cosine annealing

### Things unlikely to help (avoid wasting budget)

- Modes higher than 20 for 64×64 resolution
- Batch sizes smaller than 8 (too noisy) or larger than 64 (diminishing returns)
- Very deep models (N_LAYERS > 8) without corresponding width reduction
- Removing `add_grid` — it almost always hurts

## Experimentation

Each experiment runs on a single GPU. Training runs for a **fixed TIME_BUDGET** (defined in `prepare.py`, default 600 seconds / 10 minutes). You run it as:

```bash
python train.py > run.log 2>&1
```

**What you CAN do:**
* Modify `train.py` — this is the only file you edit. Everything is fair game: architecture via `build_model()`, the hyperparameter block, the loss function, the optimizer, the scheduler, the training loop itself.

**What you CANNOT do:**
* Modify `prepare.py`. It is read-only.
* Change `evaluate_rel_l2`. It is the ground truth metric.
* Install new packages beyond what is already in `pyproject.toml`.
* Change `T_IN`, `T_OUT`, `RESOLUTION`, `N_TRAIN`, `N_TEST` — these are fixed for comparability.

**The goal**: get the lowest `val_rel_l2`. Lower is better. Beat 0.1770.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 improvement that adds 40 lines of hacky code is not worth it. A 0.001 improvement from *deleting* code? Keep it. An improvement of ~0 but much cleaner code? Keep. Weigh complexity cost against improvement magnitude.

**VRAM**: Soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up.

## Output format

When the script finishes, it prints a summary:

```
---
val_rel_l2:        0.171234
training_seconds:  601.3
num_params_M:      2.36
num_epochs:        47
effective_batch:   64
modes:             [16, 16, 8]
n_layers:          4
mid_channels:      64
use_moe:           True
n_experts:         3
top_k:             2
---
```

Extract the key metric:

```bash
grep "^val_rel_l2:" run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:

```
commit	val_rel_l2	params_M	status	description
```

1. git commit hash (short, 7 chars)
2. val_rel_l2 achieved (e.g. `0.171234`) — use `1.000000` for crashes
3. parameter count in millions, 2 decimal places (e.g. `2.36`)
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	val_rel_l2	params_M	status	description
a1b2c3d	0.182341	1.18	keep	baseline MoEFNO modes=[16,16,8] mid=64
b2c3d4e	0.171203	4.52	keep	mid_channels 64→128
c3d4e5f	0.174900	4.52	discard	SiLU activation, worse than GELU
d4e5f6g	1.000000	0.00	crash	modes=[32,32,16] OOM
e5f6g7h	0.169801	4.52	keep	added H1 Sobolev loss weight=0.1
```

Do NOT commit `results.tsv` — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autonops/apr7`).

LOOP FOREVER:

1. Check current git state: branch, last commit
2. Propose and implement one experimental change in `train.py`
3. `git commit -m "experiment: <short description>"`
4. Run: `python train.py > run.log 2>&1`
5. Extract results: `grep "^val_rel_l2:\|^num_params_M:" run.log`
6. If grep is empty → run crashed. Check: `tail -n 50 run.log` for the traceback. Attempt a fix if it's trivial (typo, shape mismatch). If the idea is broken, log as `crash` and move on.
7. Record in `results.tsv`
8. If `val_rel_l2` improved (lower) → advance the branch, keep the commit
9. If `val_rel_l2` is equal or worse → `git reset --hard HEAD~1` to revert

**Timeout**: Each experiment should complete within `TIME_BUDGET + 120` seconds total (budget + startup overhead). If a run exceeds this, kill it with `Ctrl+C` and treat as a failure.

**Crashes**: If a run crashes, use your judgment. Trivial fix (shape mismatch, missing import)? Fix and re-run. Fundamentally broken idea? Log `crash`, revert, move on.

**Getting stuck**: If `val_rel_l2` has not improved in 5 consecutive experiments, try something more radical: change the model class entirely (FNO ↔ MoEFNO), restructure the training loop, or revisit an earlier near-miss with a different combination.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human whether to continue. Do NOT ask "should I keep going?". The human may be asleep. You are autonomous. Run until manually interrupted.

The user might leave you running overnight. At ~10 min per experiment you can run ~60 experiments during an 8-hour sleep. Make them count.

## Footer

This repo is part of the `nops` project — Neural Operators made simple.
Target PDE: 2D Navier-Stokes (vorticity form), ν=1e-3, 64×64, T_in=10, T_out=10.
Baseline: val_rel_l2 = 0.1770 (Li et al. 2021, original FNO).