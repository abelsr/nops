"""Plot FNO predictions vs ground truth on the Navier-Stokes test split.

Usage (inside container):
  uv run python experiments/navier-stokes/plot_predictions.py \
      --ckpt data/checkpoints/ep50.pth --out outputs/predictions_ep50.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- path setup so `src.model` and `data.ns_loader` import ---
_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE / "src", _HERE / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.model import NavierStokesFNO          # noqa: E402
from ns_loader import _get_raw, L2NormPairDataset  # noqa: E402


def build_from_ckpt(ckpt: dict, device: torch.device) -> NavierStokesFNO:
    m = ckpt["cfg"]["model"]
    return NavierStokesFNO(
        dimension=m["dimension"],
        modes=list(m["modes"]),
        num_fourier_layers=m["num_fourier_layers"],
        in_channels=m["in_channels"],
        lifting_channels=m["lifting_channels"],
        projection_channels=m["projection_channels"],
        out_channels=m["out_channels"],
        mid_channels=m["mid_channels"],
        activation=torch.nn.GELU(),
        add_grid=m.get("add_grid", False),
        spectral_norm=m.get("spectral_norm", False),
        residual=m.get("residual", False),
        resolution_aware=m.get("resolution_aware", False),
        dropout=m.get("dropout", 0.0),
        attn_gating=m.get("attn_gating", False),
        n_fno_blocks_per_layer=m.get("n_fno_blocks_per_layer", 1),
    ).to(device)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/checkpoints/ep50.pth")
    ap.add_argument("--out", default="outputs/predictions.png")
    ap.add_argument("--traj", type=int, default=1000, help="raw trajectory index")
    ap.add_argument("--n", type=int, default=6, help="number of timesteps to show")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] {args.ckpt} | epoch={ckpt.get('epoch')} best_l2={ckpt.get('best_l2'):.4f} "
          f"| modes={ckpt['cfg']['model']['modes']} | device={device}")

    raw = _get_raw()                     # [1200, 64, 64, 20]
    traj = raw[args.traj]                # [64, 64, 20]
    n_steps = traj.shape[-1] - 1         # 19 transitions (t -> t+1)
    n_ctx = int(ckpt["cfg"]["model"].get("in_channels", 1))

    # sample evenly spaced transitions (need n_ctx frames of history)
    ts = np.linspace(max(1, n_ctx), n_steps, args.n).astype(int)   # target t+1

    cols = len(ts)
    fig, axes = plt.subplots(3, cols, figsize=(2.6 * cols, 8.4), constrained_layout=True)
    if cols == 1:
        axes = axes.reshape(3, 1)

    for j, t_tgt in enumerate(ts):
        tgt = traj[:, :, t_tgt]
        eps = 1e-12

        # Build the model input: either a single frame or a K-frame history
        if n_ctx > 1:
            win = traj[:, :, t_tgt - n_ctx:t_tgt]           # [64, 64, K]
            n_ic = float(np.sqrt(np.mean(win ** 2)))
            x = torch.from_numpy(
                np.transpose(win, (2, 0, 1)) / (n_ic + eps)
            ).float().to(device)[None]                           # [1, K, 64, 64]
        else:
            ic = traj[:, :, t_tgt - 1]
            n_ic = float(np.sqrt(np.mean(ic ** 2)))
            x = torch.from_numpy(ic / (n_ic + eps)).float().to(device)[None, None]

        n_tgt = float(np.sqrt(np.mean(tgt ** 2)))
        pred_n = model(x).squeeze().float().cpu().numpy()

        # denormalize: model predicts the *normalized* next field
        pred = pred_n * n_tgt
        rel_l2 = np.linalg.norm(tgt - pred) / (np.linalg.norm(tgt) + 1e-10)

        vmin, vmax = tgt.min(), tgt.max()
        err = np.abs(tgt - pred)

        for row, (data, title, vlim, cmap) in enumerate([
            (tgt,  f"ground truth  t={t_tgt}",            (vmin, vmax), "RdBu_r"),
            (pred, f"prediction  (relL2={rel_l2:.3f})",   (vmin, vmax), "RdBu_r"),
            (err,  f"|error|  max={err.max():.3f}",       (0, err.max()), "magma"),
        ]):
            ax = axes[row, j]
            im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    model_desc = (f"modes={ckpt['cfg']['model']['modes']} mid={ckpt['cfg']['model']['mid_channels']} "
                  f"L={ckpt['cfg']['model']['num_fourier_layers']}")
    fig.suptitle(f"NS FNO — {model_desc} | epoch {ckpt.get('epoch')} | best_val_l2={ckpt.get('best_l2'):.4f} "
                 f"| traj {args.traj}", fontsize=11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[SAVED] {out.resolve()}")


if __name__ == "__main__":
    main()
