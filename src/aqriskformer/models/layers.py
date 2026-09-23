from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


class TemporalScaleEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
        patch: int,
        feedforward_expansion: int,
        causal_convolution: bool = False,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.causal_convolution = causal_convolution
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * feedforward_expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B,N,L,D]
        b, n, length, d = tokens.shape
        x = tokens.permute(0, 1, 3, 2).reshape(b * n, d, length)
        if self.patch > 1:
            remainder = length % self.patch
            if remainder:
                x = F.pad(x, (self.patch - remainder, 0), mode="replicate")
            x = F.avg_pool1d(x, kernel_size=self.patch, stride=self.patch)
        x = x.transpose(1, 2)
        encoded = self.encoder(x, mask=causal_mask(x.shape[1], x.device), is_causal=True)
        if self.causal_convolution:
            # Four left pads for kernel=3, dilation=2; no access to later tokens.
            conv = F.conv1d(
                F.pad(encoded.transpose(1, 2), (4, 0)),
                self.depthwise.weight,
                self.depthwise.bias,
                dilation=2,
                groups=encoded.shape[-1],
            )
        else:
            conv = self.depthwise(encoded.transpose(1, 2))[..., : encoded.shape[1]]
        conv = self.pointwise(F.gelu(conv)).transpose(1, 2)
        encoded = self.norm(encoded + conv)
        sequence = encoded.reshape(b, n, encoded.shape[1], d)
        return sequence, sequence[:, :, -1]


class AdaptiveSparseSpatialMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_stations: int,
        rank: int,
        top_k: int,
        graph_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_stations = n_stations
        self.top_k = min(top_k, n_stations)
        self.source = nn.Parameter(torch.randn(n_stations, rank) / math.sqrt(rank))
        self.target = nn.Parameter(torch.randn(n_stations, rank) / math.sqrt(rank))
        self.dynamic_q = nn.Linear(d_model, rank, bias=False)
        self.dynamic_k = nn.Linear(d_model, rank, bias=False)
        self.message = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        prior = torch.zeros(n_stations, n_stations) if graph_prior is None else graph_prior
        self.register_buffer("graph_prior", prior.float())

    def adjacency(self, hidden: torch.Tensor) -> torch.Tensor:
        static = torch.relu(self.source @ self.target.T)
        dynamic = torch.einsum(
            "bnr,bmr->bnm", self.dynamic_q(hidden), self.dynamic_k(hidden)
        ) / math.sqrt(self.source.shape[1])
        scores = static.unsqueeze(0) + dynamic + self.graph_prior.unsqueeze(0)
        values, indices = torch.topk(scores, k=self.top_k, dim=-1)
        sparse = torch.full_like(scores, float("-inf"))
        sparse.scatter_(-1, indices, values)
        return torch.softmax(sparse, dim=-1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = self.adjacency(hidden)
        neighbors = torch.einsum("bnm,bmd->bnd", adjacency, self.message(hidden))
        gate = torch.sigmoid(self.gate(torch.cat([hidden, neighbors], dim=-1)))
        mixed = gate * neighbors + (1.0 - gate) * hidden
        return self.norm(hidden + mixed), adjacency


class MissingnessAwareTokenizer(nn.Module):
    def __init__(
        self,
        n_features: int,
        calendar_dim: int,
        static_dim: int,
        n_stations: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, d_model)
        self.value_projection = nn.Linear(n_features, d_model)
        self.context_projection = nn.Linear(n_features * 2 + calendar_dim + static_dim, d_model)
        self.reliability_gate = nn.Linear(n_features * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        gaps: torch.Tensor,
        calendar: torch.Tensor,
        station_static: torch.Tensor,
        use_missingness: bool = True,
    ) -> torch.Tensor:
        _, length, n, _ = values.shape
        calendar = calendar[:, :, None, :].expand(-1, -1, n, -1)
        static = station_static[:, None, :, :].expand(-1, length, -1, -1)
        if use_missingness:
            missing_context = torch.cat([mask, gaps], dim=-1)
        else:
            missing_context = torch.cat([torch.ones_like(mask), torch.zeros_like(gaps)], dim=-1)
        context = torch.cat([missing_context, calendar, static], dim=-1)
        gate = torch.sigmoid(self.reliability_gate(missing_context))
        token = gate * self.value_projection(values) + self.context_projection(context)
        station_ids = torch.arange(n, device=values.device)
        token = token + self.station_embedding(station_ids)[None, None, :, :]
        return self.norm(token)
