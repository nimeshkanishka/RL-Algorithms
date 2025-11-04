import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [64, 64]
    ):
        super(QNetwork, self).__init__()

        layers = []        
        last_dim = input_dim
        # Hidden layers with ReLU activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        # Output layer with no activation
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)