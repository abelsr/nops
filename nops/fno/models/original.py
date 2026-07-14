from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nops.fno.layers import FourierBlock


class FNO(nn.Module):
    """
    FNO (Fourier Neural Operator) for solving PDEs.

    Features (all optional):
    - Multi-Frequency Input (MFI) for cross-resolution generalisation
    - Spectral-normalised spectral convolutions
    - Residual (identity skip) connections inside each Fourier block
    - GroupNorm on lifting layer
    - Attention gating over parallel Fourier branches
    """

    def __init__(
        self,
        modes: List[int],
        num_fourier_layers: int,
        in_channels: int,
        lifting_channels: int,
        projection_channels: int,
        out_channels: int,
        mid_channels: int,
        activation: nn.Module,
        **kwargs: Any,
    ):
        super().__init__()
        self.modes = modes
        self.dim = len(modes)
        self.num_fourier_layers = num_fourier_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.activation = activation
        self.add_grid = kwargs.get("add_grid", False)
        self.padding = kwargs.get("padding", None)
        self.n_fno_blocks_per_layer = kwargs.get("n_fno_blocks_per_layer", 2)
        self.dropout = kwargs.get("dropout", 0.0)
        self.attn_gating = kwargs.get("attn_gating", False)
        self.attn_temp = kwargs.get("attn_temperature", 1.0)
        self.spectral_norm = kwargs.get("spectral_norm", False)
        self.residual = kwargs.get("residual", False)
        self.resolution_aware = kwargs.get("resolution_aware", True)
        self.sizes = [0] * self.dim

        # --- Padding ---
        if self.padding is not None:
            self.padding = [(0, 0), (0, 0)] + [(p, p) for p in self.padding]
            self.padding = sum(self.padding, ())
            self.slice = tuple(
                slice(p, -p) if p > 0 else slice(None) for p in self.padding[2::2]
            )

        # --- Resolution-aware MFI encoding ---
        if self.resolution_aware:
            res_enc_dim = 8
            res_norm = nn.GroupNorm(4, res_enc_dim)
            self.res_encoder = nn.Sequential(
                nn.Linear(1, res_enc_dim),
                nn.SiLU(),
                nn.Linear(res_enc_dim, res_enc_dim),
            )
            self.res_norm = res_norm
            effective_in = in_channels + (self.dim if self.add_grid else 0) + res_enc_dim
        else:
            effective_in = in_channels + (self.dim if self.add_grid else 0)

        # --- Lifting (P) with GroupNorm ---
        lift_ch = self.lifting_channels or self.mid_channels
        groups = max(1, min(4, lift_ch))
        while lift_ch % groups != 0:
            groups -= 1
        if self.lifting_channels is not None:
            self.p1 = nn.Linear(effective_in, self.lifting_channels)
            if self.dim == 2:
                self.p1_norm = nn.GroupNorm(groups, self.lifting_channels)
            elif self.dim == 3:
                self.p1_norm = nn.GroupNorm(groups, self.lifting_channels)
            else:
                self.p1_norm = nn.GroupNorm(groups, self.lifting_channels)
            self.p2 = nn.Linear(self.lifting_channels, self.mid_channels)
        else:
            self.p1 = nn.Linear(effective_in, self.mid_channels)
            self.p1_norm = None
            self.p2 = None

        # --- Fourier blocks ---
        self.fourier_blocks = nn.ModuleList([
            nn.ModuleList([
                FourierBlock(
                    modes,
                    self.mid_channels,
                    self.mid_channels,
                    hidden_size=self.mid_channels,
                    activation=activation,
                    spectral_norm=self.spectral_norm,
                    residual=self.residual,
                )
                for _ in range(self.n_fno_blocks_per_layer)
            ])
            for _ in range(self.num_fourier_layers)
        ])

        # --- Dropout ---
        if self.dropout > 0.0:
            self.dropout_layer = nn.Dropout(self.dropout)

        # --- Attention gating ---
        if self.attn_gating:
            self.attn_scorer = nn.Linear(self.mid_channels, 1)

        # --- Projection (Q) with GroupNorm ---
        proj_ch = self.projection_channels or self.mid_channels
        self.q1 = nn.Linear(self.mid_channels, proj_ch)
        # Ensure groups divides channels
        groups = max(1, min(4, proj_ch))
        while proj_ch % groups != 0:
            groups -= 1
        if self.dim == 2:
            self.q1_norm = nn.GroupNorm(groups, proj_ch)
        elif self.dim == 3:
            self.q1_norm = nn.GroupNorm(groups, proj_ch)
        else:
            self.q1_norm = nn.GroupNorm(groups, proj_ch)
        self.final = nn.Linear(proj_ch, self.out_channels)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _attention_over_branches(self, Y: torch.Tensor) -> torch.Tensor:
        """Y: [K, B, C, *S] -> alpha: [B, K]"""
        if Y.dim() >= 4:
            reduce_dims = tuple(range(3, Y.dim()))
            pooled = Y.mean(dim=reduce_dims)
        else:
            pooled = Y
        KB, C = pooled.shape[0] * pooled.shape[1], pooled.shape[2]
        logits = self.attn_scorer(pooled.reshape(KB, C))
        logits = logits.reshape(pooled.shape[0], pooled.shape[1])
        logits = logits.transpose(0, 1)
        if self.attn_temp is not None and self.attn_temp > 0:
            logits = logits / self.attn_temp
        return F.softmax(logits, dim=-1)

    def _compute_resolution_encoding(
        self, x: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        """Encode resolution as a scalar feature [B, res_enc_dim]."""
        with torch.no_grad():
            log_res = torch.full(
                (x.size(0), 1),
                -4.0 + 4.0 * target_size / 1024.0,
                device=x.device,
                dtype=x.dtype,
            )
        enc = self.res_encoder(log_res)  # [B, res_enc_dim]
        enc = self.res_norm(enc.unsqueeze(2)).squeeze(2)  # GroupNorm needs (N,C,D)
        return enc  # [B, res_enc_dim]

    def _set_grid(self, x: torch.Tensor) -> None:
        batch, *sizes, _ = x.size()
        self.grids = []
        self.sizes = sizes
        for d in range(self.dim):
            new_shape = [1] * (self.dim + 2)
            new_shape[d + 1] = sizes[d]
            repeats = [1] + sizes + [1]
            repeats[d + 1] = 1
            repeats[0] = batch
            grid = (
                torch.linspace(0, 1, sizes[d], device=x.device, dtype=torch.float)
                .reshape(*new_shape)
                .repeat(repeats)
            )
            self.grids.append(grid)
        self.grids = torch.cat(self.grids, dim=-1).to(x.device)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape [batch, channels, *spatial_sizes].
            When resolution_aware=True the model will encode the max
            spatial dimension as a scalar feature so the same weights
            generalise across resolutions (MFI).

        Returns
        -------
        Tensor of shape [batch, out_channels, *spatial_sizes].
        """
        batch, _in, *sizes = x.size()
        assert len(sizes) == self.dim

        # --- Permute to [B, *S, C] ---
        x = x.permute(0, *range(2, self.dim + 2), 1)

        # --- Grid (if enabled) ---
        if self.add_grid:
            for i in range(len(sizes)):
                if sizes[i] != self.sizes[i] or (
                    hasattr(self, "grids") and self.grids.shape[0] != batch
                ):
                    self._set_grid(x)
                    break
            x = torch.cat((x, self.grids), dim=-1)

        # --- Resolution encoding (MFI) ---
        if self.resolution_aware:
            max_res = max(sizes)
            res_feat = self._compute_resolution_encoding(x, max_res)  # [B, D]
            # Reshape to [B, 1, 1, D] then expand to [B, H, W, D]
            if self.dim == 2:
                res_feat_exp = res_feat.unsqueeze(1).unsqueeze(2).expand(-1, sizes[0], sizes[1], -1)
            elif self.dim == 3:
                res_feat_exp = res_feat.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, sizes[0], sizes[1], sizes[2], -1)
            else:
                res_feat_exp = res_feat.unsqueeze(1)
            x = torch.cat([x, res_feat_exp], dim=-1)

        # --- Lifting ---
        x = self.p1(x)
        if self.p1_norm is not None:
            x = x.permute(0, -1, *range(1, self.dim + 1))
            x = self.p1_norm(x)
            x = x.permute(0, *range(2, self.dim + 2), 1)
        if self.p2 is not None:
            x = self.p2(x)

        x = x.permute(0, -1, *range(1, self.dim + 1))  # [B, C, *S]

        # --- Padding ---
        if self.padding is not None:
            x = F.pad(x, self.padding[::-1])

        # --- Fourier blocks ---
        for fourier_block in self.fourier_blocks:
            ys = [fb(x) for fb in fourier_block]
            Y = torch.stack(ys, dim=0)  # [K, B, C, *S]
            if self.attn_gating:
                alpha = self._attention_over_branches(Y)
                YB = Y.permute(1, 0, 2, *range(3, Y.dim()))
                expand_shape = [alpha.shape[0], alpha.shape[1]] + [1] * (YB.dim() - 2)
                x = (YB * alpha.view(*expand_shape)).sum(dim=1)
            else:
                x = Y.sum(dim=0)
            if self.dropout > 0.0:
                x = self.dropout_layer(x)

        # --- Remove padding ---
        if self.padding is not None:
            x = x[(Ellipsis,) + tuple(self.slice)]

        # --- Projection ---
        x = x.permute(0, *range(2, self.dim + 2), 1)  # [B, C, *S]
        x = self.q1(x)  # [B, *S, projection_channels] (Linear on last dim)

        # GroupNorm needs [B, C, *S] format, so permute
        if self.dim == 2:
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            x = self.q1_norm(x)
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        elif self.dim == 3:
            x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
            x = self.q1_norm(x)
            x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        else:
            x = x.permute(0, 2, 1)  # [B, C, L]
            x = self.q1_norm(x)
            x = x.permute(0, 2, 1)  # [B, L, C]
        x = self.activation(x)
        x = self.final(x)

        return x.permute(0, -1, *range(1, self.dim + 1))
