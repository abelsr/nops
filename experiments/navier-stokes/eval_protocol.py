"""Evaluation-protocol check for the Navier-Stokes FNO.

Computes several metric definitions on the SAME predictions so we can
quantify how much of the reported gap to the original FNO paper is due to
metric convention rather than model quality.

Variants
--------
1. per-sample mean   : mean_i  ||p_i - t_i||_2 / ||t_i||_2      (our current metric)
2. per-batch mean    : mean_b  ||P_b - T_b||_F / ||T_b||_F      (FNO repo style)
3. global ratio      : ||P - T||_F / ||T||_F  over the whole set
4. rollout (10-step) : autoregressive multi-step error in normalized space

Usage (inside container):
  uv run python experiments/navier-stokes/eval_protocol.py \
      --ckpt data/ctx_long/checkpoints/ep250.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "src", _HERE / "data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.model import NavierStokesFNO          # noqa: E402
from ns_loader import _get_raw                 # noqa: E402

EPS = 1e-12


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
        native_spectral_conv=m.get("native_spectral_conv", False),
    ).to(device)


def test_indices(n_all: int = 1200, n_train: int = 1000, n_val: int = 100, n_test: int = 100):
    """Same split as ns_loader.make_dataloaders(_ctx)."""
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_all)
    return sorted(perm[n_train + n_val:n_train + n_val + n_test].tolist())


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ctx_long/checkpoints/ep250.pth")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_ctx = int(ckpt["cfg"]["model"].get("in_channels", 1))
    ts_mode = str(ckpt["cfg"].get("training", {}).get("target_scale", "own"))
    raw = _get_raw()                      # [1200, 64, 64, 20]
    T = raw.shape[-1]
    idx_te = test_indices()
    print(f"[INFO] {args.ckpt} | n_ctx={n_ctx} | target_scale={ts_mode} | "
          f"test traj={len(idx_te)} | device={device}")

    # ------------------------------------------------------------------
    # Build the single-step test set: window -> next frame (normalized)
    # ------------------------------------------------------------------
    windows, targets, scales = [], [], []
    for ti in idx_te:
        traj = raw[ti]
        for t in range(n_ctx - 1, T - 1):
            win = traj[:, :, t - n_ctx + 1:t + 1]
            nxt = traj[:, :, t + 1]
            n_ic = float(np.sqrt(np.mean(win ** 2)))
            n_tg = float(np.sqrt(np.mean(nxt ** 2)))
            windows.append(np.transpose(win, (2, 0, 1)) / (n_ic + EPS))   # [K,64,64]
            if ts_mode == "window":
                # target in the SAME units as the input window (no oracle)
                targets.append(nxt / (n_ic + EPS))
                scales.append(n_ic)
            else:
                targets.append(nxt / (n_tg + EPS))
                scales.append(n_tg)

    W = torch.from_numpy(np.stack(windows)).float()    # [N,K,64,64]
    Y = torch.from_numpy(np.stack(targets)).float()    # [N,64,64]
    S = torch.tensor(scales).float()                   # [N] physical target norms
    N = len(W)

    preds = []
    for i in range(0, N, args.batch):
        x = W[i:i + args.batch].to(device)
        p = model(x)
        if p.shape[1] == 1:
            p = p[:, 0]
        preds.append(p.float().cpu())
    P = torch.cat(preds, dim=0)                        # [N,64,64]
    assert P.shape == Y.shape, (P.shape, Y.shape)

    # ------------------------------------------------------------------
    # 1. per-sample mean  (current metric)
    # ------------------------------------------------------------------
    per_sample = (P - Y).flatten(1).norm(dim=1) / (Y.flatten(1).norm(dim=1) + EPS)
    m_persample = per_sample.mean().item()

    # ------------------------------------------------------------------
    # 2. per-batch ratio (FNO repo style)
    # ------------------------------------------------------------------
    ratios = []
    for i in range(0, N, args.batch):
        p = P[i:i + args.batch].flatten()
        y = Y[i:i + args.batch].flatten()
        ratios.append(((p - y).norm() / (y.norm() + EPS)).item())
    m_perbatch = float(np.mean(ratios))

    # ------------------------------------------------------------------
    # 3. global ratio over the whole test set
    # ------------------------------------------------------------------
    m_global = ((P - Y).norm() / (Y.norm() + EPS)).item()

    # ------------------------------------------------------------------
    # 4. physical-domain global ratio (weights samples by n_tgt^2)
    # ------------------------------------------------------------------
    Pp = P * S[:, None, None]
    Yp = Y * S[:, None, None]
    m_global_phys = ((Pp - Yp).norm() / (Yp.norm() + EPS)).item()

    # ------------------------------------------------------------------
    # 5. scale-ablation (ONLY meaningful for target_scale="own")
    #    Replaces the true target norm n_tgt with a scale predicted from the
    #    input window.  Under target_scale="window" the model already
    #    predicts the scale, so this variant is not applicable.
    # ------------------------------------------------------------------
    n_win = []
    for ti in idx_te:
        traj = raw[ti]
        for t in range(n_ctx - 1, T - 1):
            win = traj[:, :, t - n_ctx + 1:t + 1]
            n_win.append(float(np.sqrt(np.mean(win ** 2))))
    n_win_t = torch.tensor(n_win).float()
    if ts_mode == "window":
        m_scale_err, scale_ratio = None, (S / n_win_t).mean().item()
    else:
        P_est = P * n_win_t[:, None, None]
        m_scale_err = ((P_est - Yp).norm() / (Yp.norm() + EPS)).item()
        scale_ratio = (S / n_win_t).mean().item()

    # ------------------------------------------------------------------
    # 5. autoregressive 10-step rollout in normalized space
    # ------------------------------------------------------------------
    steps = T - n_ctx                                   # 10 predicted steps
    step_err = np.zeros(steps)
    n_traj = len(idx_te)
    for k, ti in enumerate(idx_te):
        traj = raw[ti]
        # seed with the first n_ctx frames in PHYSICAL units (matches physics)
        frames = [traj[:, :, t].copy() for t in range(n_ctx)]
        for s in range(steps):
            win = np.stack(frames[-n_ctx:])                 # [K,64,64] physical
            n_win = float(np.sqrt(np.mean(win ** 2))) + EPS
            # normalise the whole window by one scalar — exactly as in training
            x = torch.from_numpy(win / n_win).float().to(device)[None]
            with torch.no_grad():
                p = model(x)
            if p.shape[1] == 1:
                p = p[:, 0]
            out = p[0].float().cpu().numpy()
            # model predicts the next frame normalised by ITS OWN norm; estimate
            # that scale from the window RMS (stationarity over the window)
            next_phys = out * n_win
            true_phys = traj[:, :, n_ctx + s]
            step_err[s] += (np.linalg.norm(next_phys - true_phys)
                            / (np.linalg.norm(true_phys) + EPS))
            frames.append(next_phys)                        # feed prediction back
    step_err /= n_traj

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  METRIC-DEFINITION COMPARISON  (same predictions)")
    print("=" * 68)
    print(f"  1. per-sample mean   (current)   : {m_persample:.4f}")
    print(f"  2. per-batch ratio   (FNO repo)  : {m_perbatch:.4f}")
    print(f"  3. global ratio      (whole set) : {m_global:.4f}")
    print(f"  4. global ratio, physical domain : {m_global_phys:.4f}")
    if m_scale_err is None:
        print(f"  5. scale ablation                : n/a "
              f"(model predicts scale; mean n_tgt/n_win = {scale_ratio:.3f})")
    else:
        print(f"  5. physical ratio w/ PREDICTED   : {m_scale_err:.4f}   "
              f"(scale from window RMS; mean n_tgt/n_win = {scale_ratio:.3f})")
    print("-" * 68)
    if ts_mode == "window":
        print("  Autoregressive rollout (oracle-free, physical units):")
    else:
        print("  Autoregressive rollout (INVALID for target_scale='own' — the")
        print("  model is scale-free, so amplitude must be guessed):")
    for s in range(steps):
        print(f"    step {s + 1:2d} (t={n_ctx + s + 1:2d}) : {step_err[s]:.4f}")
    print(f"    mean over rollout steps          : {step_err.mean():.4f}")
    print(f"    final step                       : {step_err[-1]:.4f}")
    print("=" * 68)
    print(f"  reported ckpt best_val_l2 = {ckpt.get('best_l2'):.4f}")


if __name__ == "__main__":
    main()
