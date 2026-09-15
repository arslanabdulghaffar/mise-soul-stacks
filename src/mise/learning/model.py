"""Compact ACT-style CVAE with three visual views and twelve robot joints.

This is a local implementation of the ACT architecture family, not a release of
LeRobot's ACTPolicy. It uses a shared ResNet18 visual backbone, a conditional
variational action encoder, and transformer encoder/decoder action chunking.
Reference: https://arxiv.org/abs/2304.13705
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolicyConfig:
    image_size: int = 128
    cameras: int = 3
    joints: int = 12
    chunk_size: int = 30
    hidden_dim: int = 128
    latent_dim: int = 32
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feature_grid: int = 4
    dropout: float = 0.0
    action_context: bool = False

    @property
    def state_dim(self):
        return self.joints * 2 + 1 if self.action_context else self.joints

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    """Explicit attention operators keep inference conversion framework-neutral."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Linear(dim, dim)

    def forward(self, query, memory, padding_mask=None):
        batch, length, dim = query.shape
        q = self.q(query).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(memory).reshape(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.v(memory).reshape(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], -1e4)
        values = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.out(values)


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x, padding_mask=None):
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, padding_mask)
        return x + self.ff(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.self_attention, self.cross_attention = Attention(dim, heads), Attention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x, memory):
        n = self.norm1(x)
        x = x + self.self_attention(n, n)
        x = x + self.cross_attention(self.norm2(x), memory)
        return x + self.ff(self.norm3(x))


class ContactACT(nn.Module):
    def __init__(self, config: PolicyConfig, *, pretrained_backbone: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.config = config
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        network = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        self.visual_projection = nn.Linear(512, config.hidden_dim)
        tokens = config.cameras * config.feature_grid * config.feature_grid
        self.position = nn.Parameter(torch.randn(1, tokens + 2, config.hidden_dim) * 0.02)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.latent_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.encoder = nn.ModuleList(EncoderBlock(config.hidden_dim, config.heads, config.dropout) for _ in range(config.encoder_layers))
        self.decoder = nn.ModuleList(DecoderBlock(config.hidden_dim, config.heads, config.dropout) for _ in range(config.decoder_layers))
        self.query = nn.Parameter(torch.randn(1, config.chunk_size, config.hidden_dim) * 0.02)
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.joints))
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.posterior_position = nn.Parameter(torch.randn(1, config.chunk_size + 2, config.hidden_dim) * 0.02)
        self.action_projection = nn.Linear(config.joints, config.hidden_dim)
        self.posterior = nn.ModuleList(EncoderBlock(config.hidden_dim, config.heads, config.dropout) for _ in range(2))
        self.posterior_statistics = nn.Linear(config.hidden_dim, config.latent_dim * 2)

    def visual_features(self, images):
        batch, cameras, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch * cameras, channels, height, width))
        features = F.adaptive_avg_pool2d(features, self.config.feature_grid)
        features = features.flatten(2).transpose(1, 2).reshape(batch, -1, 512)
        return self.visual_projection(features)

    def decode(self, features, state, latent):
        memory = torch.cat((self.state_projection(state).unsqueeze(1), self.latent_projection(latent).unsqueeze(1), features), dim=1)
        memory = memory + self.position
        for block in self.encoder:
            memory = block(memory)
        query = self.query.expand(state.shape[0], -1, -1)
        for block in self.decoder:
            query = block(query, memory)
        return self.output(query)

    def forward(self, images, state):
        """Deterministic prior inference; only RGB and proprioception enter."""
        latent = torch.zeros(state.shape[0], self.config.latent_dim, dtype=state.dtype, device=state.device)
        return self.decode(self.visual_features(images), state, latent)

    def training_prediction(self, images, state, actions, padding_mask):
        cls = self.posterior_cls.expand(state.shape[0], -1, -1)
        tokens = torch.cat((cls, self.state_projection(state).unsqueeze(1), self.action_projection(actions)), dim=1)
        tokens = tokens + self.posterior_position
        mask = torch.cat((torch.zeros(state.shape[0], 2, dtype=torch.bool, device=state.device), padding_mask), dim=1)
        for block in self.posterior:
            tokens = block(tokens, mask)
        mean, log_variance = self.posterior_statistics(tokens[:, 0]).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-10, 10)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        prediction = self.decode(self.visual_features(images), state, latent)
        kl = (-0.5 * (1 + log_variance - mean.square() - log_variance.exp())).sum(dim=-1).mean()
        return prediction, kl
