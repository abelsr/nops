"""
prepare.py — Dataset, evaluación y constantes fijas
=====================================================
Este archivo NO se modifica. Contiene:
  - Constantes del experimento (TIME_BUDGET, resolución, splits, etc.)
  - Carga y preprocesamiento del dataset de Navier-Stokes
  - evaluate_rel_l2(): la función de evaluación canónica

El dataset esperado es un archivo .pt con la key "u" de shape:
    (N_samples, H, W, T_total)
donde T_total >= T_IN + T_OUT.

Descarga el dataset original de Li et al. (2021) desde:
    https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
Archivo recomendado: NavierStokes_V1e-3_N1200_T20.mat  (ν=1e-3, 1200 muestras, 20 timesteps)

Alternativamente, ajusta DATA_PATH a tu .pt/.npz/.npy local.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# CONSTANTES — NO MODIFICAR
# =============================================================================

# Tiempo fijo de entrenamiento (wall clock, excl. startup/compilación)
TIME_BUDGET = 600          # segundos (10 min — ajusta según tu GPU)

# Configuración del problema
T_IN       = 10            # timesteps de entrada
T_OUT      = 10            # timesteps de salida (a predecir)
RESOLUTION = 64            # resolución espacial H=W=64

# Splits
N_TRAIN    = 1000
N_TEST     = 200

# Path al dataset
DATA_PATH = os.environ.get(
    "NS_DATA_PATH",
    os.path.expanduser("~/.cache/nops/navier_stokes_v1e-3_N1200_T20.pt")
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CARGA DE DATOS
# =============================================================================

def _load_raw() -> torch.Tensor:
    """
    Carga el dataset y devuelve tensor de shape (N, H, W, T_total).
    Soporta .pt (key "u"), .npz (key "u") y .npy.
    """
    path = DATA_PATH
    assert os.path.exists(path), (
        f"Dataset no encontrado en {path}.\n"
        f"Descarga NavierStokes_V1e-3_N1200_T20.mat y conviértelo, "
        f"o apunta NS_DATA_PATH a tu archivo."
    )

    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        u = data["u"] if isinstance(data, dict) else data
    elif path.endswith(".npz"):
        data = np.load(path)
        u = torch.from_numpy(data["u"].astype(np.float32))
    elif path.endswith(".npy"):
        u = torch.from_numpy(np.load(path).astype(np.float32))
    else:
        raise ValueError(f"Formato no soportado: {path}")

    # Asegurar float32
    u = u.float()

    # Esperamos (N, H, W, T)
    assert u.ndim == 4, f"Esperado shape (N,H,W,T), got {u.shape}"
    N, H, W, T = u.shape
    assert T >= T_IN + T_OUT, (
        f"El dataset tiene {T} timesteps pero se necesitan T_IN+T_OUT={T_IN+T_OUT}."
    )
    assert H == RESOLUTION and W == RESOLUTION, (
        f"Resolución esperada {RESOLUTION}x{RESOLUTION}, got {H}x{W}."
    )
    return u


def get_dataloaders(batch_size: int = 16, num_workers: int = 4):
    """
    Devuelve (train_loader, val_loader).

    Shapes de los tensores:
        x: [B, T_IN,  H, W]   — input:  primeros T_IN  timesteps
        y: [B, T_OUT, H, W]   — target: siguientes T_OUT timesteps

    BUG CORREGIDO respecto al original:
        El script original hacía u[:,:,:,:10] y u[:,:,:,10:] sobre un tensor
        con solo 10 timesteps, dejando y vacío. Aquí el split es explícito y
        verificado.
    """
    u = _load_raw()                   # (N, H, W, T_total)

    x = u[:, :, :, :T_IN]            # (N, H, W, T_IN)
    y = u[:, :, :, T_IN:T_IN+T_OUT]  # (N, H, W, T_OUT)

    # Permute a (N, T, H, W) — canales = timesteps
    x = x.permute(0, 3, 1, 2)        # (N, T_IN,  H, W)
    y = y.permute(0, 3, 1, 2)        # (N, T_OUT, H, W)

    # Normalización por channel (media y std sobre train)
    x_mean = x[:N_TRAIN].mean(dim=(0, 2, 3), keepdim=True)
    x_std  = x[:N_TRAIN].std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)
    x = (x - x_mean) / x_std

    y_mean = y[:N_TRAIN].mean(dim=(0, 2, 3), keepdim=True)
    y_std  = y[:N_TRAIN].std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)
    y = (y - y_mean) / y_std

    train_ds = TensorDataset(x[:N_TRAIN], y[:N_TRAIN])
    val_ds   = TensorDataset(x[N_TRAIN:N_TRAIN+N_TEST], y[N_TRAIN:N_TRAIN+N_TEST])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# =============================================================================
# EVALUACIÓN — función canónica, NO MODIFICAR
# =============================================================================

@torch.no_grad()
def evaluate_rel_l2(model: torch.nn.Module, val_loader: DataLoader, device) -> float:
    """
    Relative L2 error sobre el validation set.
    Métrica canónica de la literatura de Neural Operators (Li et al. 2021).
    Lower is better. Referencia FNO original en NS (ν=1e-3): ~0.1770
    """
    model.eval()
    total_num   = 0.0
    total_denom = 0.0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        b    = y.shape[0]
        total_num   += torch.norm(
            (pred - y).reshape(b, -1), dim=1
        ).sum().item()
        total_denom += torch.norm(
            y.reshape(b, -1), dim=1
        ).sum().item()

    model.train()
    return total_num / max(total_denom, 1e-8)