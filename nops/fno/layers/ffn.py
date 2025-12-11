from typing import Literal

from torch import nn

class FeedForwardNet(nn.Module):
    """
    
    FeedForwardNet is a simple feed-forward neural network with two hidden layers.
    It supports configurable hidden layer sizes, dropout, and choice of non-linearity.
    
    Args:
        input_dim (int): Dimension of the input features.
        hidden1 (int): Number of units in the first hidden layer. Default is 128.
        hidden2 (int): Number of units in the second hidden layer. Default is 64.
        output_dim (int): Dimension of the output features. Default is 1.
        dropout (float): Dropout rate applied after each hidden layer. Default is 0.1.
        non_linearity (str): Type of non-linearity to use ('relu' or 'gelu'). Default is 'relu'.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden1: int = 128, 
        hidden2: int = 64, 
        output_dim: int = 1,
        dropout: float = 0.1,
        non_linearity: Literal['relu', 'gelu'] = 'relu'
    ):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU() if non_linearity == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU() if non_linearity == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.model(x)
