"""
train.py — Navier-Stokes Neural Operator training script
=========================================================
Este es el archivo que el agente modifica. Todo es fair game:
arquitectura, optimizador, hiperparámetros, loss, batch size, etc.

El archivo que NO se toca es prepare.py, que contiene:
  - Carga del dataset
  - La función de evaluación: evaluate_rel_l2()
  - Constantes fijas (TIME_BUDGET, T_IN, T_OUT, RESOLUTION)

Métrica objetivo: val_rel_l2  (relative L2 error, lower is better)
Referencia FNO original en NS (ν=1e-3): ~0.1770

Tiempo de run: fijo en TIME_BUDGET segundos (ver prepare.py).
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from prepare import (
    get_dataloaders,
    evaluate_rel_l2,
    TIME_BUDGET,   # segundos de entrenamiento (wall clock, excl. startup)
    T_IN,          # timesteps de entrada  (e.g. 10)
    T_OUT,         # timesteps de salida   (e.g. 10)
    RESOLUTION,    # resolución espacial   (e.g. 64)
    DEVICE,
)

torch.set_float32_matmul_precision("medium")

# =============================================================================
# HIPERPARÁMETROS — el agente modifica esta sección
# =============================================================================

# Datos
BATCH_SIZE = 16          # batch por GPU
ACCUM_STEPS = 4          # gradient accumulation → batch efectivo = BATCH_SIZE * ACCUM_STEPS

# Arquitectura  (FNO 2D: x, y) - temporal dimension handled as channels
MODES      = [16, 16]      # modos de Fourier por dimensión [x, y] - 2D spatial only
N_LAYERS   = 4             # número de Fourier/MoE layers
MID_CH     = 128           # canales intermedios (width del modelo) - Increased from 64
LIFT_CH    = 128           # canales de lifting
PROJ_CH    = 128           # canales de projection
ADD_GRID   = True          # agregar coordenadas espaciales al input

# Si usas MoEFNO
USE_MOE        = False
N_EXPERTS      = 3
EXPERT_HIDDEN  = 32
TOP_K          = 2
ROUTING_TYPE   = "patch"   # "patch" | "sample"

# Optimizador
LR           = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_GRAD    = 1.0

# Scheduler: warmup lineal + cosine annealing
# (el agente puede cambiar esto a OneCycle, StepLR, etc.)
WARMUP_STEPS = 500   # pasos de warmup (no épocas, para ser agnóstico al dataset)

# =============================================================================
# LOSS — relative L2 (estándar en literatura de Neural Operators)
# =============================================================================

def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Relative L2 loss por muestra, promediada sobre el batch.
    pred, target: [B, C, *sizes]
    """
    diff   = pred - target
    b      = diff.shape[0]
    num    = torch.norm(diff.reshape(b, -1), dim=1)
    denom  = torch.norm(target.reshape(b, -1), dim=1).clamp(min=1e-8)
    return (num / denom).mean()

# =============================================================================
# MODELO — el agente puede cambiar entre FNO y MoEFNO, o crear variantes
# =============================================================================

def build_model() -> nn.Module:
    """
    Construye el modelo. El agente puede cambiar esta función libremente.

    IMPORTANTE sobre in_channels cuando ADD_GRID=True:
      - Input raw:  [B, T_IN, H, W]  →  in_channels = T_IN   (sin grid)
      - Con grid 3D (x,y,t):          →  in_channels efectivo = T_IN + 3
      Pasamos in_channels=T_IN y dejamos que add_grid maneje el resto internamente.
      MoEFNO suma self.dim=3 canales de grid en su lifting, así que
      lifting_in = in_channels + dim = T_IN + 3  ✓
    """
    if USE_MOE:
        from nops.fno.models.moe_fno import MoEFNO
        model = MoEFNO(
            modes=MODES,
            num_moe_layers=N_LAYERS,
            num_experts=N_EXPERTS,
            in_channels=T_IN,          # ← T_IN, NO T_IN+3 (MoEFNO suma grid internamente)
            lifting_channels=LIFT_CH,
            projection_channels=PROJ_CH,
            out_channels=T_OUT,
            mid_channels=MID_CH,
            expert_hidden_size=EXPERT_HIDDEN,
            top_k=TOP_K,
            routing_type=ROUTING_TYPE,
            add_grid=ADD_GRID,
        )
    else:
        from nops.fno.models.original import FNO
        model = FNO(
            modes=MODES,
            num_fourier_layers=N_LAYERS,
            in_channels=T_IN,
            lifting_channels=LIFT_CH,
            projection_channels=PROJ_CH,
            out_channels=T_OUT,
            mid_channels=MID_CH,
            activation=nn.GELU(),
            add_grid=ADD_GRID,
        )
    return model.to(DEVICE)


# =============================================================================
# SCHEDULER
# =============================================================================

def build_scheduler(optimizer, total_steps: int):
    """
    OneCycleLR scheduler: fast warmup then cosine annealing.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 3,
        total_steps=total_steps,
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos'
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model     = build_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=True
    )
    scaler = GradScaler()

    # Estimación de steps totales para el scheduler
    # (heurística: asumimos ~3 épocas completas en el budget)
    steps_per_epoch  = len(train_loader) // ACCUM_STEPS
    estimated_epochs = max(1, TIME_BUDGET // (steps_per_epoch * 2 + 1))
    total_steps      = steps_per_epoch * estimated_epochs
    scheduler = build_scheduler(optimizer, total_steps)

    # ── Loop ──────────────────────────────────────────────────────────────────
    start_time   = time.time()
    global_step  = 0
    epoch        = 0
    optimizer.zero_grad()

    while True:
        # ── Comprobación de tiempo ────────────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        epoch += 1
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            elapsed = time.time() - start_time
            if elapsed >= TIME_BUDGET:
                break

            x = x.to(DEVICE, non_blocking=True)   # [B, T_IN,  H, W]
            y = y.to(DEVICE, non_blocking=True)    # [B, T_OUT, H, W]

            with autocast():
                pred = model(x)
                loss = rel_l2_loss(pred, y) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        # ── Validación al final de cada época ────────────────────────────────
        val_rel_l2 = evaluate_rel_l2(model, val_loader, DEVICE)
        elapsed    = time.time() - start_time
        print(
            f"epoch: {epoch:4d} | "
            f"val_rel_l2: {val_rel_l2:.6f} | "
            f"elapsed: {elapsed:.1f}s | "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

    # ── Resumen final ─────────────────────────────────────────────────────────
    training_seconds = time.time() - start_time
    val_rel_l2_final = evaluate_rel_l2(model, val_loader, DEVICE)
    n_params         = sum(p.numel() for p in model.parameters())

    print("\n---")
    print(f"val_rel_l2:        {val_rel_l2_final:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"num_params_M:      {n_params / 1e6:.2f}")
    print(f"num_epochs:        {epoch}")
    print(f"effective_batch:   {BATCH_SIZE * ACCUM_STEPS}")
    print(f"modes:             {MODES}")
    print(f"n_layers:          {N_LAYERS}")
    print(f"mid_channels:      {MID_CH}")
    print(f"use_moe:           {USE_MOE}")
    if USE_MOE:
        print(f"n_experts:         {N_EXPERTS}")
        print(f"top_k:             {TOP_K}")
    print("---")


if __name__ == "__main__":
    train()