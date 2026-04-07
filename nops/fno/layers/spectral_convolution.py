from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.amp as amp
import torch.utils.checkpoint as checkpoint

import tensorly as tl
from tensorly.decomposition import tucker, parafac, tensor_train

# Set TensorLy to use PyTorch as the backend
tl.set_backend('pytorch')


class SpectralConvolution(nn.Module):
    """
    Spectral Convolution layer optimized with support for tensor factorization,
    mixed-precision training, and N-dimensional data.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (List[int]): List of modes for spectral convolution in each dimension.
        factorization (str, optional): Type of factorization to use ('dense', 'tucker', 'cp', 'tt').
                                       Defaults to 'tucker'.
        rank (int, optional): Rank for low-rank factorization. Defaults to 8.
        bias (bool, optional): Whether to include a bias term in the layer. Defaults to True.
        **kwargs: Additional parameters.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: List[int],
        factorization: str = 'tucker',
        rank: Union[int, List[int]] = 8,
        bias: bool = True,
        use_orthogonal: bool = True,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = len(self.modes)
        self.factorization = factorization.lower()
        self.use_orthogonal = use_orthogonal
        
        # Handle adaptive rank
        if isinstance(rank, int):
            self.rank = [rank] * (2 + self.dim)
        else:
            if len(rank) != 2 + self.dim:
                raise ValueError(f"Rank list must have length {2 + self.dim}")
            self.rank = rank

        # Validate factorization type
        if self.factorization not in ['dense', 'tucker', 'cp', 'tt']:
            raise ValueError("Unsupported factorization. Choose from 'dense', 'tucker', 'cp', 'tt'.")

        # Generate the mixing matrix
        self.mix_matrix = self.get_mix_matrix(self.dim)

        # Weight factorization based on selected type
        if self.factorization == 'dense':
            # Full weights without factorization
            weight_shape = (in_channels, out_channels, *self.modes)
            self.weights_real = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(weight_shape, dtype=torch.float32))
            )
            self.weights_imag = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(weight_shape, dtype=torch.float32))
            )
        elif self.factorization == 'tucker':
            # Initialize Tucker factors with orthogonal initialization
            factor_shapes = [in_channels, out_channels] + self.modes
            core_shape = self.rank
            
            self.core_real = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(core_shape, dtype=torch.float32))
            )
            self.core_imag = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(core_shape, dtype=torch.float32))
            )
            
            factors_real = []
            factors_imag = []
            for i, shape in enumerate(factor_shapes):
                if self.use_orthogonal and shape >= self.rank[i]:
                    # Initialize with orthogonal matrices
                    factor_r = nn.init.orthogonal_(torch.empty(shape, self.rank[i], dtype=torch.float32))
                    factor_i = nn.init.orthogonal_(torch.empty(shape, self.rank[i], dtype=torch.float32))
                else:
                    factor_r = nn.init.xavier_uniform_(torch.empty(shape, self.rank[i], dtype=torch.float32))
                    factor_i = nn.init.xavier_uniform_(torch.empty(shape, self.rank[i], dtype=torch.float32))
                factors_real.append(nn.Parameter(factor_r))
                factors_imag.append(nn.Parameter(factor_i))
            
            self.factors_real = nn.ParameterList(factors_real)
            self.factors_imag = nn.ParameterList(factors_imag)
        elif self.factorization == 'cp':
            # Initialize CP factors with improved initialization
            factor_shapes = [in_channels, out_channels] + self.modes
            rank_scalar = self.rank[0] if isinstance(self.rank, list) else self.rank
            
            # Initialize weights with uniform distribution instead of ones
            self.weights_cp_real = nn.Parameter(
                torch.rand(rank_scalar, dtype=torch.float32) * 0.02 - 0.01
            )
            self.weights_cp_imag = nn.Parameter(
                torch.rand(rank_scalar, dtype=torch.float32) * 0.02 - 0.01
            )
            
            factors_cp_real = []
            factors_cp_imag = []
            for shape in factor_shapes:
                if self.use_orthogonal and shape >= rank_scalar:
                    factor_r = nn.init.orthogonal_(torch.empty(shape, rank_scalar, dtype=torch.float32))
                    factor_i = nn.init.orthogonal_(torch.empty(shape, rank_scalar, dtype=torch.float32))
                else:
                    factor_r = nn.init.xavier_uniform_(torch.empty(shape, rank_scalar, dtype=torch.float32))
                    factor_i = nn.init.xavier_uniform_(torch.empty(shape, rank_scalar, dtype=torch.float32))
                factors_cp_real.append(nn.Parameter(factor_r))
                factors_cp_imag.append(nn.Parameter(factor_i))
            
            self.factors_cp_real = nn.ParameterList(factors_cp_real)
            self.factors_cp_imag = nn.ParameterList(factors_cp_imag)
        elif self.factorization == 'tt':
            # Initialize TT cores with improved stability
            factor_shapes = [in_channels, out_channels] + self.modes
            rank_scalar = self.rank[0] if isinstance(self.rank, list) else self.rank
            
            tt_cores_real = []
            tt_cores_imag = []
            for i, shape in enumerate(factor_shapes):
                left_rank = 1 if i == 0 else rank_scalar
                right_rank = 1 if i == len(factor_shapes) - 1 else rank_scalar
                core_shape = (left_rank, shape, right_rank)
                
                # Scale initialization for stability
                scale = 1.0 / (left_rank * right_rank) ** 0.5
                core_r = nn.init.uniform_(torch.empty(core_shape, dtype=torch.float32), -scale, scale)
                core_i = nn.init.uniform_(torch.empty(core_shape, dtype=torch.float32), -scale, scale)
                
                tt_cores_real.append(nn.Parameter(core_r))
                tt_cores_imag.append(nn.Parameter(core_i))
            
            self.factors_tt_real = nn.ParameterList(tt_cores_real)
            self.factors_tt_imag = nn.ParameterList(tt_cores_imag)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None
    
    def _tucker_mult_direct(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        slices: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tucker multiplication using partial reconstruction.
        Only reconstructs the needed spectral modes, not the full tensor.
        """
        # Get sliced factors for spatial dimensions
        factors_real_sliced = [self.factors_real[0], self.factors_real[1]]
        factors_imag_sliced = [self.factors_imag[0], self.factors_imag[1]]
        
        for i, s in enumerate(slices):
            # Slice the spatial factors to only the needed modes
            factors_real_sliced.append(self.factors_real[2 + i][s, :])
            factors_imag_sliced.append(self.factors_imag[2 + i][s, :])
        
        # Reconstruct only the sliced weights
        weights_real = tl.tucker_to_tensor((self.core_real, factors_real_sliced))
        weights_imag = tl.tucker_to_tensor((self.core_imag, factors_imag_sliced))
        
        # Perform complex multiplication
        x_sliced_real = x_real[(...,) + slices]
        x_sliced_imag = x_imag[(...,) + slices]
        
        return self.complex_mult(x_sliced_real, x_sliced_imag, weights_real, weights_imag)
    
    def _cp_mult_direct(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        slices: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CP multiplication using partial reconstruction.
        Only reconstructs the needed spectral modes.
        """
        # Get sliced factors for spatial dimensions
        factors_real_sliced = [self.factors_cp_real[0], self.factors_cp_real[1]]
        factors_imag_sliced = [self.factors_cp_imag[0], self.factors_cp_imag[1]]
        
        for i, s in enumerate(slices):
            factors_real_sliced.append(self.factors_cp_real[2 + i][s, :])
            factors_imag_sliced.append(self.factors_cp_imag[2 + i][s, :])
        
        # Reconstruct only the sliced weights
        weights_real = tl.cp_to_tensor((self.weights_cp_real, factors_real_sliced))
        weights_imag = tl.cp_to_tensor((self.weights_cp_imag, factors_imag_sliced))
        
        # Perform complex multiplication
        x_sliced_real = x_real[(...,) + slices]
        x_sliced_imag = x_imag[(...,) + slices]
        
        return self.complex_mult(x_sliced_real, x_sliced_imag, weights_real, weights_imag)
    
    def _get_weights_cached(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazy reconstruction with caching for dense operations.
        Only reconstructs when needed and caches the result.
        """
        if not hasattr(self, '_cached_weights_real'):
            if self.factorization == 'tucker':
                self._cached_weights_real = tl.tucker_to_tensor(
                    (self.core_real, [f for f in self.factors_real])
                )
                self._cached_weights_imag = tl.tucker_to_tensor(
                    (self.core_imag, [f for f in self.factors_imag])
                )
            elif self.factorization == 'cp':
                self._cached_weights_real = tl.cp_to_tensor(
                    (self.weights_cp_real, [f for f in self.factors_cp_real])
                )
                self._cached_weights_imag = tl.cp_to_tensor(
                    (self.weights_cp_imag, [f for f in self.factors_cp_imag])
                )
            elif self.factorization == 'tt':
                self._cached_weights_real = tl.tt_to_tensor(self.factors_tt_real)
                self._cached_weights_imag = tl.tt_to_tensor(self.factors_tt_imag)
        
        return self._cached_weights_real, self._cached_weights_imag
    
    def _clear_cache(self):
        """Clear cached weights (call during training step)"""
        if hasattr(self, '_cached_weights_real'):
            del self._cached_weights_real
            del self._cached_weights_imag
    
    def orthogonalize_factors(self):
        """
        Orthogonalize factor matrices using QR decomposition.
        Call this periodically during training for numerical stability.
        
        Note: For Tucker with adaptive ranks, we only normalize factors
        without absorbing into core to avoid dimension mismatch issues.
        """
        if not self.use_orthogonal:
            return
        
        with torch.no_grad():
            if self.factorization == 'tucker':
                # Check if ranks are uniform
                ranks_uniform = len(set(self.rank)) == 1
                
                if ranks_uniform:
                    # Full orthogonalization with R absorption
                    for i, (factor_r, factor_i) in enumerate(zip(self.factors_real, self.factors_imag)):
                        if factor_r.shape[0] >= factor_r.shape[1]:
                            # QR decomposition
                            q_r, r_r = torch.linalg.qr(factor_r)
                            q_i, r_i = torch.linalg.qr(factor_i)
                            
                            # Update factor
                            factor_r.copy_(q_r)
                            factor_i.copy_(q_i)
                            
                            # Absorb R into core along dimension i
                            core_new_real = torch.tensordot(self.core_real.data, r_r, dims=([i], [1]))
                            core_new_imag = torch.tensordot(self.core_imag.data, r_i, dims=([i], [1]))
                            self.core_real.data = core_new_real
                            self.core_imag.data = core_new_imag
                else:
                    # Adaptive ranks: just orthogonalize without core absorption
                    for factor_r, factor_i in zip(self.factors_real, self.factors_imag):
                        if factor_r.shape[0] >= factor_r.shape[1]:
                            q_r, _ = torch.linalg.qr(factor_r)
                            q_i, _ = torch.linalg.qr(factor_i)
                            factor_r.copy_(q_r)
                            factor_i.copy_(q_i)
            
            elif self.factorization == 'cp':
                for factor_r, factor_i in zip(self.factors_cp_real, self.factors_cp_imag):
                    if factor_r.shape[0] >= factor_r.shape[1]:
                        # Normalize columns
                        norms_r = torch.norm(factor_r, dim=0, keepdim=True)
                        norms_i = torch.norm(factor_i, dim=0, keepdim=True)
                        
                        factor_r.div_(norms_r + 1e-8)
                        factor_i.div_(norms_i + 1e-8)
                        
                        # Absorb norms into weights
                        self.weights_cp_real.data.mul_(norms_r.squeeze())
                        self.weights_cp_imag.data.mul_(norms_i.squeeze())
        
        # Clear cache after orthogonalization
        self._clear_cache()
    
    def get_compression_ratio(self) -> float:
        """
        Calculate the compression ratio of the factorization.
        
        Returns:
            float: Ratio of factorized parameters to full tensor parameters.
        """
        full_size = self.in_channels * self.out_channels * torch.prod(torch.tensor(self.modes)).item()
        
        if self.factorization == 'dense':
            return 1.0
        elif self.factorization == 'tucker':
            factorized_size = torch.prod(torch.tensor(self.rank)).item()
            for i, shape in enumerate([self.in_channels, self.out_channels] + self.modes):
                factorized_size += shape * self.rank[i]
        elif self.factorization == 'cp':
            rank_scalar = self.rank[0] if isinstance(self.rank, list) else self.rank
            factorized_size = rank_scalar  # weights
            for shape in [self.in_channels, self.out_channels] + self.modes:
                factorized_size += shape * rank_scalar
        elif self.factorization == 'tt':
            rank_scalar = self.rank[0] if isinstance(self.rank, list) else self.rank
            shapes = [self.in_channels, self.out_channels] + self.modes
            factorized_size = shapes[0] * rank_scalar  # first core
            for i in range(1, len(shapes) - 1):
                factorized_size += rank_scalar * shapes[i] * rank_scalar
            factorized_size += rank_scalar * shapes[-1]  # last core
        else:
            return 1.0
        
        return factorized_size / full_size

    @staticmethod
    def complex_mult(input_real: torch.Tensor, input_imag: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs complex multiplication between input and weights.

        Args:
            input_real (torch.Tensor): Real part of the input. [batch_size, in_channels, *sizes]
            input_imag (torch.Tensor): Imaginary part of the input. [batch_size, in_channels, *sizes]
            weights_real (torch.Tensor): Real part of the weights. [in_channels, out_channels, *sizes]
            weights_imag (torch.Tensor): Imaginary part of the weights. [in_channels, out_channels, *sizes]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts of the result. [batch_size, out_channels, *sizes]
        """
        out_real = torch.einsum('bi...,io...->bo...', input_real, weights_real) - torch.einsum('bi...,io...->bo...', input_imag, weights_imag)
        out_imag = torch.einsum('bi...,io...->bo...', input_real, weights_imag) + torch.einsum('bi...,io...->bo...', input_imag, weights_real)
        return out_real, out_imag

    @staticmethod
    def get_mix_matrix(dim: int) -> torch.Tensor:
        """
        Generates a mixing matrix for spectral convolution.

        Args:
            dim (int): Dimension of the mixing matrix.

        Returns:
            torch.Tensor: Mixing matrix.

        The mixing matrix is generated in the following steps:
        1. Create a lower triangular matrix filled with ones and subtract 2 times the identity matrix to introduce negative values.
        2. Subtract 2 from the last row to ensure a distinct pattern for mixing.
        3. Set the last element of the last row to 1 to maintain a consistent matrix structure.
        4. Convert all zero elements to 1, ensuring no zero values are present.
        5. Add a row of ones at the beginning to provide an additional mixing row.
        """
        # Step 1: Create a lower triangular matrix with -1 on the diagonal and 1 elsewhere
        mix_matrix = torch.tril(torch.ones((dim, dim), dtype=torch.float32)) - 2 * torch.eye(dim, dtype=torch.float32)

        # Step 2: Subtract 2 from the last row
        mix_matrix[-1] = mix_matrix[-1] - 2

        # Step 3: Set the last element of the last row to 1
        mix_matrix[-1, -1] = 1

        # Step 4: Convert zeros in the mixing matrix to 1
        mix_matrix[mix_matrix == 0] = 1

        # Step 5: Add a row of ones at the beginning
        mix_matrix = torch.cat((torch.ones((1, dim), dtype=torch.float32), mix_matrix), dim=0)

        return mix_matrix

    def mix_weights(
        self,
        out_ft_real: torch.Tensor,
        out_ft_imag: torch.Tensor,
        x_ft_real: torch.Tensor,
        x_ft_imag: torch.Tensor,
        weights_real: Union[List[torch.Tensor], torch.Tensor],
        weights_imag: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixes weights for spectral convolution.

        Args:
            out_ft_real (torch.Tensor): Real part of the output tensor in Fourier space.
            out_ft_imag (torch.Tensor): Imaginary part of the output tensor in Fourier space.
            x_ft_real (torch.Tensor): Real part of the input tensor in Fourier space.
            x_ft_imag (torch.Tensor): Imaginary part of the input tensor in Fourier space.
            weights_real (List[torch.Tensor] or torch.Tensor): Real weights.
            weights_imag (List[torch.Tensor] or torch.Tensor): Imaginary weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mixed output tensors (real and imaginary parts).
        """
        # Slicing indices based on the mixing matrix
        slices = tuple(slice(None, min(mode, x_ft_real.size(i + 2))) for i, mode in enumerate(self.modes))

        # Mix weights
        # First weight
        out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
            x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices],
            weights_real[(Ellipsis,) + slices], weights_imag[(Ellipsis,) + slices] # type: ignore
        )

        if isinstance(weights_real, list) and len(weights_real) > 1:
            # Remaining weights
            for i in range(1, len(weights_real)):
                modes = self.mix_matrix[i].squeeze().tolist()
                slices = tuple(
                    slice(-min(mode, x_ft_real.size(j + 2)), None) if sign < 0 else slice(None, min(mode, x_ft_real.size(j + 2)))
                    for j, (sign, mode) in enumerate(zip(modes, self.modes))
                )
                out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
                    x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices],
                    weights_real[i][(Ellipsis,) + slices], weights_imag[i][(Ellipsis,) + slices]
                )

        return out_ft_real, out_ft_imag

    def forward(self, x: torch.Tensor, use_direct_mult: bool = True) -> torch.Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D1, D2, ..., DN).
            use_direct_mult (bool): Use direct factorized multiplication (faster, less memory).
                                   If False, reconstructs full tensor (slower, more memory).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, D1, D2, ..., DN).
        """
        batch_size, _, *sizes = x.shape

        # Ensure input has the expected number of dimensions
        if len(sizes) != self.dim:
            raise ValueError(f"Expected input to have {self.dim + 2} dimensions (including batch and channel), but got {len(sizes) + 2}")

        # Apply N-dimensional FFT in float32 for numerical stability
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with amp.autocast(device_type, enabled=False): # type: ignore
            x_ft = torch.fft.fftn(x.float(), dim=tuple(range(-self.dim, 0)), norm='ortho')

        # Separate into real and imaginary parts
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag

        # Initialize output tensors in Fourier space
        out_ft_real = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_real.dtype, device=x.device)
        out_ft_imag = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_imag.dtype, device=x.device)

        # Determine slices for first mode combination
        slices = tuple(slice(None, min(mode, x_ft_real.size(i + 2))) for i, mode in enumerate(self.modes))

        # Apply factorization-specific computation
        if self.factorization == 'dense':
            # Use weights directly
            out_ft_real, out_ft_imag = self.mix_weights(
                out_ft_real, out_ft_imag, x_ft_real, x_ft_imag, self.weights_real, self.weights_imag
            )
        elif self.factorization == 'tucker':
            if use_direct_mult:
                # Direct Tucker multiplication without full reconstruction
                out_ft_real[(...,) + slices], out_ft_imag[(...,) + slices] = self._tucker_mult_direct(
                    x_ft_real, x_ft_imag, slices
                )
            else:
                # Fallback: reconstruct and use mix_weights
                weights_real, weights_imag = self._get_weights_cached()
                out_ft_real, out_ft_imag = self.mix_weights(
                    out_ft_real, out_ft_imag, x_ft_real, x_ft_imag,
                    weights_real, weights_imag
                )
        elif self.factorization == 'cp':
            if use_direct_mult:
                # Direct CP multiplication
                out_ft_real[(...,) + slices], out_ft_imag[(...,) + slices] = self._cp_mult_direct(
                    x_ft_real, x_ft_imag, slices
                )
            else:
                # Fallback: reconstruct and use mix_weights
                weights_real, weights_imag = self._get_weights_cached()
                out_ft_real, out_ft_imag = self.mix_weights(
                    out_ft_real, out_ft_imag, x_ft_real, x_ft_imag,
                    weights_real, weights_imag
                )
        elif self.factorization == 'tt':
            # TT always reconstructs (TensorLy's tt_to_tensor is already optimized)
            # But we use caching to avoid repeated reconstruction
            weights_real, weights_imag = self._get_weights_cached()
            out_ft_real, out_ft_imag = self.mix_weights(
                out_ft_real, out_ft_imag, x_ft_real, x_ft_imag,
                weights_real, weights_imag
            )

        # Combine real and imaginary parts
        out_ft = torch.complex(out_ft_real, out_ft_imag)

        # Apply IFFT to return to spatial domain
        out = torch.fft.ifftn(out_ft, dim=tuple(range(-self.dim, 0)), s=sizes, norm='ortho').real

        # Add learnable bias if present
        if self.bias is not None:
            bias = self.bias.view(1, -1, *([1] * self.dim))
            out = out + bias

        return out
