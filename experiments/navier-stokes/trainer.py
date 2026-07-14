"""
Navier-Stokes FNO trainer.

Usage
-----
  uv run python experiments/navier-stokes/trainer.py \\
    'model.modes=[16,16]' \\
    model.mid_channels=128 \\
    model.num_fourier_layers=8 \\
    exp_name=my_exp \\
    training.lr=0.001 training.epochs=200
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# ---------------------------------------------------------------------------
# Helpers: metrics, activation resolver
# ---------------------------------------------------------------------------


def resolve_activation(name: str) -> nn.Module:
    map = {"nn.GELU": nn.GELU, "nn.SiLU": nn.SiLU, "nn.ReLU": nn.ReLU, "nn.Tanh": nn.Tanh}
    return map.get(name, lambda: nn.GELU())()


@torch.no_grad()
def relative_l2(true: torch.Tensor, pred: torch.Tensor) -> float:
    return (true - pred).float().norm(p=2).item() / (true.float().norm(p=2) + 1e-10)


@torch.no_grad()
def relative_l1(true: torch.Tensor, pred: torch.Tensor) -> float:
    return (true.float() - pred.float()).abs().mean().item() / (true.abs().float().mean() + 1e-10)


@torch.no_grad()
def mean_energy_error(true: torch.Tensor, pred: torch.Tensor) -> float:
    e_true = 0.5 * true.float().pow(2).mean(dim=(1, 2))
    e_pred = 0.5 * pred.float().pow(2).mean(dim=(1, 2))
    return (e_pred - e_true).abs().mean().item() / (e_true.mean().abs() + 1e-10)


# ---------------------------------------------------------------------------
# Compact logger — prints 1–2 lines per epoch + final summary
# ---------------------------------------------------------------------------


class CompactLogger:
    """Token-efficient logger.  Prints:
    
      [EP]  t_l2=0.XX v_l2=0.XX v_l1=X.XX v_E=X.XX | best=0.XX(epN) lr=X.Xe-X
    
    and a final summary block.  Training loss is aggregated per epoch (no step-by-step).
    """

    def __init__(self, exp_name: str) -> None:
        self.exp_name = exp_name
        self.history: list[dict] = []
        self.best_l2 = float("inf")
        self.best_ep = 0

    # -- public API ----------------------------------------------------------

    def log_epoch(self, epoch: int, train_l2: float, val_l2: float, val_l1: float,
                  val_E: float, lr: float) -> None:
        if val_l2 < self.best_l2:
            self.best_l2 = val_l2
            self.best_ep = epoch

        self.history.append({
            "ep": epoch, "tr_l2": train_l2, "v_l2": val_l2,
            "v_l1": val_l1, "v_E": val_E, "lr": lr,
        })

        # One compact line per epoch
        print(
            f"[EP] {self._fmt_ep(epoch, 4)} "
            f"t_l2={train_l2:.4f} v_l2={val_l2:.4f} v_l1={val_l1:.4f} v_E={val_E:.4f} "
            f"| b={self.best_l2:.4f}(ep{self.best_ep}) lr={lr:.2e}",
            flush=True,
        )

    def finish(self, t0: float, test: dict, n_params: int, cfg: DictConfig) -> dict:
        """Final summary + JSON output.  Called once at the very end."""
        h = self.history
        last = h[-1] if h else {}

        # Compact ASCII chart (last 20 epochs, scaled to width 40)
        chart_lines = self._make_chart(h[-30:] if len(h) > 30 else h)

        lines = [
            "",
            "─────────────────────────────────────────────────────────────",
            f"  {self.exp_name or 'ns_nop'}  |  {len(h)} ep | {time.time()-t0:.0f}s | {n_params:,} params",
            f"  best  v_l2 = {self.best_l2:.4f}  (epoch {self.best_ep})",
            f"  test  v_l2 = {test.get('l2', float('nan')):.4f}   v_l1 = {test.get('l1', float('nan')):.4f}   v_E = {test.get('energy_err', float('nan')):.4f}",
        ]
        if chart_lines:
            lines += chart_lines

        lines.append("─────────────────────────────────────────────────────────────")

        print("\n".join(lines), flush=True)

        result = {
            "model": {
                "mid_channels": cfg.model.mid_channels,
                "num_fourier_layers": cfg.model.num_fourier_layers,
                "modes": list(cfg.model.modes),
                "residual": cfg.model.get("residual", False),
                "spectral_norm": cfg.model.get("spectral_norm", False),
                "resolution_aware": cfg.model.get("resolution_aware", True),
            },
            "training": {
                "epochs": len(h),
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "train_samples": cfg.training.train_samples,
                "val_samples": cfg.training.val_samples,
                "test_samples": cfg.training.test_samples,
            },
            "result": {
                "best_val_l2": round(self.best_l2, 4),
                "best_val_l2_epoch": self.best_ep,
                "test_l2": round(test.get("l2", 0), 4),
                "test_l1": round(test.get("l1", 0), 4),
                "test_energy_err": round(test.get("energy_err", 0), 4),
                "num_parameters": n_params,
                "time_sec": round(time.time() - t0, 0),
            },
        }

        # Machine-parseable block
        print(f"===JSON==={json.dumps(result)}==END_JSON===", flush=True)

        return result

    # -- internals -----------------------------------------------------------

    def _fmt_ep(self, ep: int, total: int) -> str:
        w = max(4, len(str(total)))
        return f"{ep:>{w}}/{total}"

    @staticmethod
    def _make_chart(history: list[dict]) -> list[str]:
        """Mini bar-chart comparing last N epoch val_l2 values."""
        if not history:
            return []
        vals = [h["v_l2"] for h in history]
        mn, mx = min(vals), max(vals)
        span = mx - mn or 1
        width = 24
        lines = [f"  {'val_l2 trend (last '+str(len(history))+')':<{width}}"]
        for h in history:
            bar = "█" * max(1, int((h["v_l2"] - mn) / span * width))
            ep_w = len(str(h["ep"]))
            lines.append(f"  ep{h['ep']:<{ep_w + 1}} |{bar} {h['v_l2']:.4f}")
        return lines


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> nn.Module:
    import sys
    from pathlib import Path as _P
    _d = _P(__file__).resolve().parent
    if str(_d.parent) not in sys.path:
        sys.path.insert(0, str(_d.parent))
    from src.model import NavierStokesFNO

    act_s = getattr(cfg.model, "activation", "nn.GELU")
    add_grid = cfg.model.get("add_grid", True)
    act = resolve_activation(act_s) if isinstance(act_s, str) else nn.GELU()

    return NavierStokesFNO(
        dimension=cfg.model.dimension,
        modes=list(cfg.model.modes),
        num_fourier_layers=cfg.model.num_fourier_layers,
        in_channels=cfg.model.in_channels,
        lifting_channels=cfg.model.lifting_channels,
        projection_channels=cfg.model.projection_channels,
        out_channels=cfg.model.out_channels,
        mid_channels=cfg.model.mid_channels,
        activation=act,
        add_grid=add_grid,
        spectral_norm=cfg.model.get("spectral_norm", False),
        residual=cfg.model.get("residual", False),
        resolution_aware=cfg.model.get("resolution_aware", False),
        dropout=cfg.model.get("dropout", 0.0),
        attn_gating=cfg.model.get("attn_gating", False),
        n_fno_blocks_per_layer=cfg.model.get("n_fno_blocks_per_layer", 1),
    )


# ---------------------------------------------------------------------------
# LRScheduler helper
# ---------------------------------------------------------------------------

def make_scheduler(sched: str, optim_: optim.Optimizer, warmup: int,
                   min_lr: float, max_lr: float, warmup_lr: float,
                   epochs: int):
    warmup_factor = warmup_lr / max_lr if max_lr > 0 else 0.0
    warm_sched = LinearLR(optim_, start_factor=warmup_factor, end_factor=1.0,
                          total_iters=warmup) if warmup > 0 else None

    t_max = max(1, epochs - warmup)
    cos_sched = CosineAnnealingLR(optim_, T_max=t_max, eta_min=min_lr) if sched == "cosine" else None

    if warm_sched and cos_sched:
        return SequentialLR(optim_, [warm_sched, cos_sched], milestones=[warmup])
    return warm_sched or cos_sched


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, max_samples: int | None = None) -> Dict[str, float]:
    """Evaluate metrics on normalized data (same scale as training)."""
    model.eval()
    l2_s, l1_s, e_s, n = 0.0, 0.0, 0.0, 0
    t0 = time.time()
    for batch in loader:
        ic = batch["vorticity_ic"].to(device)
        tgt = batch["vorticity"].to(device)
        pred = model(ic.unsqueeze(1)).squeeze(1)
        if pred.shape != tgt.shape:
            pred = nn.functional.interpolate(pred.unsqueeze(1), size=tgt.shape[1:],
                                              mode="bilinear", align_corners=False).squeeze(1)
        l2_s += relative_l2(tgt, pred) * ic.size(0)
        l1_s += relative_l1(tgt, pred) * ic.size(0)
        e_s += mean_energy_error(tgt, pred) * ic.size(0)
        n += ic.size(0)
        if max_samples and n >= max_samples:
            break

    # Also compute normalized energy error (using L2 norms as reference)
    # This gives a scale-independent energy metric
    if max_samples and n > 0:
        # subsampled energy: just compute from first batch to save time
        pass

    return {
        "l2": float(l2_s / (n + 1e-10)), "l1": float(l1_s / (n + 1e-10)),
        "energy_err": float(e_s / (n + 1e-10)), "time": time.time() - t0,
        "n": n,
    }
    return {
        "l2": l2_s / (n + 1e-10), "l1": l1_s / (n + 1e-10),
        "energy_err": e_s / (n + 1e-10), "time": time.time() - t0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "configs" / "v1"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] {device} | seed={cfg.seed}", flush=True)

    # --- Model ---
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] {n_params:,} params  config = {OmegaConf.to_yaml(cfg.model)}", flush=True)

    # --- Data ---
    import sys
    _dd = Path(__file__).resolve().parent / "data"
    if str(_dd) not in sys.path:
        sys.path.insert(0, str(_dd))
    from ns_loader import make_dataloaders

    loaders = make_dataloaders(
        n_train=cfg.training.train_samples,
        n_val=cfg.training.val_samples,
        n_test=cfg.training.test_samples,
        batch_size=cfg.training.batch_size,
        val_batch_size=cfg.training.val_batch_size,
        device=device,
    )
    print(f"[INIT] train={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}  test={len(loaders['test'].dataset)}", flush=True)

    # --- Optimiser / Scheduler ---
    optim_ = optim.AdamW(model.parameters(), lr=cfg.training.lr,
                         weight_decay=cfg.training.weight_decay)
    sched = make_scheduler(cfg.training.get("scheduler", "cosine"), optim_,
                           cfg.training.warmup_epochs,
                           cfg.training.get("min_lr", 1e-6),
                           cfg.training.lr,
                           cfg.training.get("warmup_lr", 1e-6),
                           cfg.training.epochs)

    # --- Resume ---
    start_ep = 0
    if (ckpt := cfg.training.get("resume")) and Path(ckpt).exists():
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"])
        optim_.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"])
        start_ep = c["epoch"] + 1
        print(f"[INIT] resume epoch={c['epoch']}  best_l2={c['best_l2']:.4f}", flush=True)

    # --- Training loop ---
    logger = CompactLogger(cfg.get("exp_name", "") or "ns_nop")
    best_l2 = logger.best_l2
    best_state = None
    patience = cfg.training.get("early_stopping_patience")
    patience_cnt = 0
    total_steps = 0

    print(f"\n[TRAIN] {cfg.training.epochs} ep | bs={cfg.training.batch_size}  "
          f"lr={cfg.training.lr:.2e}  wd={cfg.training.weight_decay}", flush=True)

    t0 = time.time()

    for ep in range(start_ep, cfg.training.epochs):
        optim_.param_groups[0]["lr"] = sched.get_last_lr()[0]
        model.train()

        ep_loss = 0.0
        ep_n = 0

        for batch in loaders["train"]:
            ic = batch["vorticity_ic"].to(device)
            tgt = batch["vorticity"].to(device)

            pred = model(ic.unsqueeze(1)).squeeze(1)
            loss = (pred.float() - tgt.float()).pow(2).mean()

            if cfg.training.get("loss_enstrophy", 0) > 0:
                loss += cfg.training.loss_enstrophy * (pred.float().pow(2).mean())
            if cfg.training.get("loss_energy", 0) > 0:
                loss += cfg.training.loss_energy * (pred.float().pow(2).std() - tgt.float().pow(2).std()).pow(2)

            optim_.zero_grad()
            loss.backward()
            if grad_clip := cfg.training.get("grad_clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim_.step()

            ep_loss += loss.item() * ic.size(0)
            ep_n += ic.size(0)
            total_steps += 1

        sched.step()

        avg_loss = ep_loss / (ep_n + 1e-10)

        # --- Validation (every val_interval epochs) ---
        if (ep + 1) % cfg.training.get("val_interval", 1) == 0:
            vm = evaluate(model, loaders["val"], device)
            tm = evaluate(model, loaders["train"], device,
                          max_samples=200)  # subsample train for speed

            logger.log_epoch(ep + 1, tm["l2"], vm["l2"], vm["l1"], vm["energy_err"],
                             optim_.param_groups[0]["lr"])

            better = vm["l2"] < best_l2
            if better:
                best_l2 = float(vm["l2"])
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                logger.best_l2 = best_l2
                logger.best_ep = ep + 1

            # Checkpoint
            ci = cfg.training.get("checkpoint_interval", 50)
            if ci and (ep + 1) % ci == 0:
                ckpt_dir = Path(cfg.training.data_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": ep + 1, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optim_.state_dict(), "scheduler": sched.state_dict(),
                    "best_l2": best_l2, "best_epoch": logger.best_ep,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                }, ckpt_dir / f"ep{ep+1}.pth")

            if patience is not None:
                patience_cnt = 0 if better else patience_cnt + 1
                if patience_cnt >= patience:
                    print(f"  >> early-stop ep {ep+1}", flush=True)
                    break

        # Epoch-level training loss
        # (not printed — already in compact logger via train L2 eval,
        #  but we print average training loss every N epochs for debug)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  [LOSS] epoch {ep+1}  avg_mse={avg_loss:.6f}", flush=True)

    # --- Test (best model) ---
    if best_state:
        model.load_state_dict(best_state)

    test = evaluate(model, loaders["test"], device)
    result = logger.finish(t0, test, n_params, cfg)

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
