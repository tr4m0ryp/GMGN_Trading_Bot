"""
Custom feature extractors for RL trading agents.

Architectures: TradingFeaturesExtractor (MLP), AdvancedTradingFeaturesExtractor
(multi-scale + position-aware), HybridLSTMAttentionExtractor (LSTM + Attention).

Dependencies: torch, stable_baselines3, gymnasium
Author: Trading Team
Date: 2025-12-23
"""

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """MLP feature extractor: Input -> 2x (Linear+LayerNorm+GELU+Dropout) -> Output."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.network(observations)


class AdvancedTradingFeaturesExtractor(BaseFeaturesExtractor):
    """Multi-scale feature extractor with position-aware processing (last 4 dims)."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))
        hidden = features_dim // 2

        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.scale2 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        # Combine scales
        self.combine = nn.Sequential(
            nn.Linear(hidden * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Position-aware processing
        # Last 4 dims: in_position, entry_price, unrealized_pnl, time_in_position
        self.position_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Final projection
        self.output = nn.Sequential(
            nn.Linear(features_dim + 32, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        # Split features and position info
        position_info = observations[..., -4:]

        # Multi-scale processing of features
        s1 = self.scale1(observations)
        s2 = self.scale2(observations)
        combined = self.combine(torch.cat([s1, s2], dim=-1))

        # Position-aware features
        pos_features = self.position_net(position_info)

        # Combine all
        out = self.output(torch.cat([combined, pos_features], dim=-1))

        return out


class HybridLSTMAttentionExtractor(BaseFeaturesExtractor):
    """Hybrid LSTM + Multi-Head Attention extractor with position-aware fusion."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))
        self.lstm_hidden = lstm_hidden

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
        )

        # Bidirectional LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Multi-head self-attention for long-range dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(lstm_hidden * 2)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 4, lstm_hidden * 2),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(lstm_hidden * 2)

        # Position-aware processing (last 4 dims)
        self.position_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # Market feature processing (remaining dims)
        self.market_net = nn.Sequential(
            nn.Linear(input_dim - 4, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
        )

        # Final projection combining all features
        combined_dim = lstm_hidden * 2 + 64 + lstm_hidden
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
        )

        # Confidence head for selective trading
        self.confidence_head = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features using LSTM + Attention hybrid architecture.

        Args:
            observations: Input tensor of shape (batch, obs_dim)

        Returns:
            Features tensor of shape (batch, features_dim)
        """
        # Split position info and market features
        market_features = observations[..., :-4]
        position_info = observations[..., -4:]

        # Project input for LSTM
        x = self.input_proj(observations)

        # Add sequence dimension for LSTM (batch, 1, hidden)
        x = x.unsqueeze(1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + ffn_out)

        # Squeeze sequence dimension
        lstm_features = ffn_out.squeeze(1)

        # Position-aware features
        pos_features = self.position_net(position_info)

        # Market features
        mkt_features = self.market_net(market_features)

        # Combine all features
        combined = torch.cat([lstm_features, pos_features, mkt_features], dim=-1)

        # Final projection
        output = self.output_proj(combined)

        return output

    def get_confidence(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get confidence score for the extracted features.

        Args:
            features: Features tensor from forward pass

        Returns:
            Confidence scores (batch, 1) in range [0, 1]
        """
        return self.confidence_head(features)
