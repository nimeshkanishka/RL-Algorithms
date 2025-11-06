from typing import Optional
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

class DQNPolicy(nn.Module):
    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        network_architecture: list[int],
        activation_fn: type[nn.Module],
        feature_extractor_architecture: Optional[list[tuple]] = None,
        normalize_images: bool = False
    ):
        super(DQNPolicy, self).__init__()

        # Create feature extractor (CNN)
        # If feature_extractor_architecture is not specified (for MLP Policy),
        # this will be an empty nn.Sequential
        feature_extractor = []        
        if feature_extractor_architecture is not None:            
            in_channels = observation_space.shape[0]
            # Convolution layers with activations
            for layer in feature_extractor_architecture:
                feature_extractor.append(nn.Conv2d(in_channels, *layer))
                feature_extractor.append(activation_fn())
                in_channels = layer[0]
        # Always add Flatten layer to feature extractor
        # For already flat inputs (e.g. [B, N]), this is a no-op
        self.feature_extractor = nn.Sequential(
            *feature_extractor,
            nn.Flatten()
        )
        
        # Create network (MLP)
        network = []
        in_features = self._calculate_feature_extractor_out_dim(observation_space.shape)
        # Linear layers with activations
        for layer in network_architecture:
            network.append(nn.Linear(in_features, layer))
            network.append(activation_fn())
            in_features = layer
        # Output layer with no activation
        network.append(nn.Linear(in_features, action_space.n))
        self.network = nn.Sequential(*network)        

        self.normalize_images = normalize_images

    def forward(self, x):
        if self.normalize_images:
            x = x / 255.0

        return self.network(self.feature_extractor(x))
    
    def _calculate_feature_extractor_out_dim(self, observation_space_shape):
        # Dummy input with batch dimension
        dummy_input = torch.zeros(*observation_space_shape).unsqueeze(dim=0)
        # Feature extractor output shape: (1, feature_extractor_out_dim)
        return self.feature_extractor(dummy_input).shape[-1]
    
class MLPPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        network_architecture: list[int] = [64, 64],
        activation_fn: type[nn.Module] = nn.ReLU,
        feature_extractor_architecture: Optional[list[tuple]] = None,
        normalize_images: Optional[bool] = None
    ):
        super(MLPPolicy, self).__init__(
            observation_space,
            action_space,
            network_architecture,
            activation_fn,
            feature_extractor_architecture
        )

class CNNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        network_architecture: list[int] = [512],
        activation_fn: type[nn.Module] = nn.ReLU,
        feature_extractor_architecture: list[tuple] = [
            (32, 8, 4),
            (64, 4, 2),
            (64, 3, 1)
        ],
        normalize_images: bool = True
    ):
        super(CNNPolicy, self).__init__(
            observation_space,
            action_space,
            network_architecture,
            activation_fn,
            feature_extractor_architecture,
            normalize_images
        )