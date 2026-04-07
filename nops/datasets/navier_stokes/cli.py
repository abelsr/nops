import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .generator import generate_ns_data


def _build_forcing(resolution, forcing, amplitude, device):
    if forcing == "none":
        return torch.zeros(resolution, resolution, device=device)
    if forcing == "constant":
        return amplitude * torch.ones(resolution, resolution, device=device)

    x = torch.linspace(0.0, 1.0, resolution, device=device)
    y = torch.linspace(0.0, 1.0, resolution, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    arg_x = 2.0 * math.pi * xx
    arg_y = 2.0 * math.pi * yy

    if forcing == "sin":
        return amplitude * torch.sin(arg_x) * torch.sin(arg_y)
    if forcing == "cos":
        return amplitude * torch.cos(arg_x) * torch.cos(arg_y)

    raise ValueError(f"Unsupported forcing: {forcing}")


def _resolve_output_path(output, fmt):
    output_path = Path(output)
    if output_path.exists() and output_path.is_dir():
        return output_path / f"navier_stokes.{fmt}"
    if output_path.suffix == "":
        return output_path.with_suffix(f".{fmt}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Navier-Stokes dataset samples.")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution (power of 2).")
    parser.add_argument("--num-samples", type=int, required=True, help="Total number of samples to generate.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation.")
    parser.add_argument("--visc", type=float, default=1e-3, help="Viscosity coefficient.")
    parser.add_argument("--delta-t", type=float, default=1e-4, help="Time step size.")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final simulation time.")
    parser.add_argument("--record-steps", type=int, default=10, help="Number of recorded time steps.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--forcing", choices=["none", "constant", "sin", "cos"], default="none")
    parser.add_argument("--forcing-amplitude", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="navier_stokes.pt")
    parser.add_argument("--format", choices=["pt", "mat"], default="pt")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.num_samples % args.batch_size != 0:
        parser.error("--num-samples must be divisible by --batch-size.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    f = _build_forcing(args.resolution, args.forcing, args.forcing_amplitude, device)

    start = time.time()
    a, u, t = generate_ns_data(
        resolution=args.resolution,
        N=args.num_samples,
        f=f,
        visc=args.visc,
        delta_t=args.delta_t,
        T_final=args.t_final,
        record_steps=args.record_steps,
        batch_size=args.batch_size,
        device=device,
        debug=args.debug,
    )
    duration = time.time() - start

    output_path = _resolve_output_path(args.output, args.format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "pt":
        torch.save({"a": a, "u": u, "t": t}, output_path)
    else:
        try:
            import scipy.io as sio
        except ImportError as exc:
            raise SystemExit("scipy is required for --format mat") from exc
        sio.savemat(output_path.as_posix(), {"a": a.numpy(), "u": u.numpy(), "t": t.numpy()})

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": output_path.name,
        "format": args.format,
        "resolution": args.resolution,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "visc": args.visc,
        "delta_t": args.delta_t,
        "t_final": args.t_final,
        "record_steps": args.record_steps,
        "device": str(device),
        "seed": args.seed,
        "forcing": args.forcing,
        "forcing_amplitude": args.forcing_amplitude,
        "a_shape": list(a.shape),
        "u_shape": list(u.shape),
        "t_shape": list(t.shape),
        "duration_seconds": round(duration, 6),
    }
    metadata_path = output_path.with_name("metadata.json")
    with metadata_path.open("w", encoding="utf-8") as fobj:
        json.dump(metadata, fobj, indent=2, sort_keys=True, ensure_ascii=True)


if __name__ == "__main__":
    main()
