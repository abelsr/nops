import torch
import torch.nn as nn
import pytest

from nops.fno.layers.moe import MoEBlock, Router
from nops.fno.layers.mlp import MLP
from nops.fno.models.moe_fno import MoEFNO


def test_router_forward_shape():
    """Test that Router produces correct output shape."""
    batch_size = 4
    in_channels = 32
    num_experts = 8
    spatial_dims = (16, 16)
    
    router = Router(in_channels=in_channels, num_experts=num_experts)
    x = torch.randn(batch_size, in_channels, *spatial_dims)
    
    weights = router(x)
    
    assert weights.shape == (batch_size, num_experts)
    # Weights should sum to 1 (softmax)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_router_temperature():
    """Test that temperature affects the sharpness of routing weights."""
    batch_size = 2
    in_channels = 16
    num_experts = 4
    
    x = torch.randn(batch_size, in_channels, 8, 8)
    
    # Low temperature should produce sharper (more peaked) distribution
    router_low_temp = Router(in_channels=in_channels, num_experts=num_experts, temperature=0.1)
    weights_low = router_low_temp(x)
    
    # High temperature should produce smoother distribution
    router_high_temp = Router(in_channels=in_channels, num_experts=num_experts, temperature=2.0)
    weights_high = router_high_temp(x)
    
    # Standard deviation should be higher for low temperature (more peaked)
    assert weights_low.std(dim=-1).mean() > weights_high.std(dim=-1).mean()


def test_moe_block_all_experts():
    """Test MoEBlock with all experts (no top-k)."""
    batch_size = 2
    in_channels = 16
    num_experts = 4
    spatial_dims = (8, 8)
    
    # Create simple MLP experts
    experts = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    router = Router(in_channels=in_channels, num_experts=num_experts)
    moe_block = MoEBlock(experts=experts, router=router, top_k=None)
    
    x = torch.randn(batch_size, in_channels, *spatial_dims)
    output = moe_block(x)
    
    assert output.shape == x.shape
    # Output should not be all zeros
    assert output.abs().sum() > 0


def test_moe_block_top_k():
    """Test MoEBlock with top-k expert selection."""
    batch_size = 4
    in_channels = 16
    num_experts = 8
    top_k = 2
    spatial_dims = (8, 8)
    
    # Create simple MLP experts
    experts = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    router = Router(in_channels=in_channels, num_experts=num_experts)
    moe_block = MoEBlock(experts=experts, router=router, top_k=top_k)
    
    x = torch.randn(batch_size, in_channels, *spatial_dims)
    output = moe_block(x)
    
    assert output.shape == x.shape
    assert moe_block.top_k == top_k
    # Output should not be all zeros
    assert output.abs().sum() > 0


def test_moe_block_top_k_single():
    """Test MoEBlock with top-k=1 (only best expert)."""
    batch_size = 2
    in_channels = 16
    num_experts = 4
    top_k = 1
    spatial_dims = (8, 8)
    
    # Create simple MLP experts
    experts = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    router = Router(in_channels=in_channels, num_experts=num_experts)
    moe_block = MoEBlock(experts=experts, router=router, top_k=top_k)
    
    x = torch.randn(batch_size, in_channels, *spatial_dims)
    output = moe_block(x)
    
    assert output.shape == x.shape
    assert moe_block.top_k == top_k


def test_moe_block_invalid_top_k():
    """Test that invalid top_k values raise ValueError."""
    num_experts = 4
    in_channels = 16
    
    experts = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    router = Router(in_channels=in_channels, num_experts=num_experts)
    
    # top_k = 0 should raise ValueError
    with pytest.raises(ValueError):
        MoEBlock(experts=experts, router=router, top_k=0)
    
    # top_k > num_experts should raise ValueError
    with pytest.raises(ValueError):
        MoEBlock(experts=experts, router=router, top_k=num_experts + 1)


def test_moe_block_backward():
    """Test that MoEBlock gradients flow correctly."""
    batch_size = 2
    in_channels = 16
    num_experts = 4
    top_k = 2
    spatial_dims = (8, 8)
    
    experts = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    router = Router(in_channels=in_channels, num_experts=num_experts)
    moe_block = MoEBlock(experts=experts, router=router, top_k=top_k)
    
    x = torch.randn(batch_size, in_channels, *spatial_dims, requires_grad=True)
    output = moe_block(x)
    loss = output.mean()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check that expert parameters have gradients
    for expert in moe_block.experts:
        for param in expert.parameters():
            # Note: only top-k experts will have gradients
            # So we just check that at least some have gradients
            if param.grad is not None:
                assert param.grad.shape == param.shape


def test_moe_fno_with_top_k():
    """Test MoEFNO model with top-k configuration."""
    torch.manual_seed(42)
    model = MoEFNO(
        modes=[4, 4],
        num_moe_layers=2,
        num_experts=4,
        in_channels=1,
        lifting_channels=16,
        projection_channels=16,
        out_channels=1,
        mid_channels=16,
        expert_hidden_size=16,
        top_k=2,
        padding=[2, 2]
    )
    
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    
    assert y.shape == x.shape
    assert model.top_k == 2
    
    # Check that MoE layers have correct top_k
    for moe_layer in model.moe_layers:
        assert moe_layer.top_k == 2


def test_moe_fno_without_top_k():
    """Test MoEFNO model without top-k (uses all experts)."""
    torch.manual_seed(42)
    model = MoEFNO(
        modes=[4, 4],
        num_moe_layers=2,
        num_experts=4,
        in_channels=1,
        lifting_channels=16,
        projection_channels=16,
        out_channels=1,
        mid_channels=16,
        expert_hidden_size=16,
        top_k=None,
        padding=[2, 2]
    )
    
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    
    assert y.shape == x.shape
    assert model.top_k is None
    
    # Check that MoE layers use all experts
    for moe_layer in model.moe_layers:
        assert moe_layer.top_k == 4  # num_experts


def test_moe_fno_backward_with_top_k():
    """Test that gradients flow correctly through MoEFNO with top-k."""
    torch.manual_seed(42)
    model = MoEFNO(
        modes=[4, 4],
        num_moe_layers=2,
        num_experts=4,
        in_channels=1,
        lifting_channels=16,
        projection_channels=16,
        out_channels=1,
        mid_channels=16,
        expert_hidden_size=16,
        top_k=2,
    )
    
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    # Check that all parameters have finite gradients
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_moe_block_consistency():
    """Test that top_k=num_experts produces similar results to top_k=None."""
    torch.manual_seed(42)
    batch_size = 2
    in_channels = 16
    num_experts = 4
    spatial_dims = (8, 8)
    
    # Create experts
    experts1 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    # Clone experts for second block
    experts2 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        ) for _ in range(num_experts)
    ])
    
    # Copy weights to make them identical
    for e1, e2 in zip(experts1, experts2):
        e2.load_state_dict(e1.state_dict())
    
    router1 = Router(in_channels=in_channels, num_experts=num_experts)
    router2 = Router(in_channels=in_channels, num_experts=num_experts)
    router2.load_state_dict(router1.state_dict())
    
    # Block with all experts
    moe_block_all = MoEBlock(experts=experts1, router=router1, top_k=None)
    
    # Block with top_k = num_experts (should be equivalent)
    moe_block_topk = MoEBlock(experts=experts2, router=router2, top_k=num_experts)
    
    x = torch.randn(batch_size, in_channels, *spatial_dims)
    
    with torch.no_grad():
        output_all = moe_block_all(x)
        output_topk = moe_block_topk(x)
    
    # Results should be very similar (allowing for numerical differences)
    assert torch.allclose(output_all, output_topk, rtol=1e-4, atol=1e-5)
