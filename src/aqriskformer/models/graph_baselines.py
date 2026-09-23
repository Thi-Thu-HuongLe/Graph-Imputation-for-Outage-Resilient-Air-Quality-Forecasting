from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import _ProbabilisticOutputMixin


def _probabilistic_output(
    mu: torch.Tensor, log_scale: torch.Tensor, stations: int
) -> dict[str, torch.Tensor]:
    sigma = F.softplus(log_scale)[None, None, None, :].expand_as(mu) + 1e-4
    return {
        "mu": mu,
        "sigma": sigma,
        "nu": torch.full_like(mu, 1e6),
        "adjacency": torch.eye(stations, device=mu.device)[None].expand(mu.shape[0], -1, -1),
    }


class DiffusionGraphLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.projection = nn.Linear(input_dim * (1 + 2 * diffusion_steps), output_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.to(dtype=x.dtype)
        features = [x]
        forward, backward = x, x
        for _ in range(self.diffusion_steps):
            forward = torch.einsum("nm,bmc->bnc", adjacency, forward)
            backward = torch.einsum("mn,bmc->bnc", adjacency, backward)
            features.extend([forward, backward])
        return self.projection(torch.cat(features, dim=-1))


class DiffusionGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionGraphLinear(input_dim + hidden, hidden * 2, diffusion_steps)
        self.candidate = DiffusionGraphLinear(input_dim + hidden, hidden, diffusion_steps)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat([x, state], dim=-1), adjacency)).chunk(
            2, dim=-1
        )
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * state], dim=-1), adjacency))
        return update * state + (1.0 - update) * candidate


class DCRNNForecast(nn.Module, _ProbabilisticOutputMixin):
    """Direct multi-horizon DCRNN baseline with train-only graph support."""

    def __init__(
        self,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        graph_prior: torch.Tensor,
        hidden: int = 64,
        layers: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        self.n_stations, self.n_pollutants, self.horizon = n_stations, n_pollutants, horizon
        adjacency = graph_prior.float() + torch.eye(n_stations)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("adjacency", adjacency)
        input_dim = n_features * 3 + 8
        self.cells = nn.ModuleList(
            [
                DiffusionGRUCell(input_dim if layer == 0 else hidden, hidden)
                for layer in range(layers)
            ]
        )
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, length, n, _ = values.shape
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        sequence = torch.cat([values, mask, gaps, calendar], dim=-1)
        states = [values.new_zeros(b, n, cell.hidden) for cell in self.cells]
        for step in range(length):
            hidden = sequence[:, step]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(hidden, states[layer], self.adjacency)
                hidden = states[layer]
        mu = self.head(states[-1]).reshape(b, n, self.horizon, self.output_width)
        mu = mu.permute(0, 2, 1, 3)
        output = self._wrap(mu)
        output["adjacency"] = self.adjacency[None].expand(b, -1, -1)
        return output


class _GraphWaveBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.filter = nn.Conv2d(
            hidden, hidden, (1, 3), dilation=(1, dilation), padding=(0, padding)
        )
        self.gate = nn.Conv2d(hidden, hidden, (1, 3), dilation=(1, dilation), padding=(0, padding))
        self.graph_projection = nn.Conv2d(hidden * 2, hidden, 1)
        self.residual = nn.Conv2d(hidden, hidden, 1)
        self.skip = nn.Conv2d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        filtered = self.filter(x)
        gated = self.gate(x)
        if self.padding:
            filtered = filtered[..., : -self.padding]
            gated = gated[..., : -self.padding]
        temporal = torch.tanh(filtered) * torch.sigmoid(gated)
        propagated = torch.einsum(
            "nm,bcmt->bcnt", adjacency.to(dtype=temporal.dtype), temporal
        )
        graph = self.graph_projection(torch.cat([temporal, propagated], dim=1))
        graph = self.dropout(graph)
        return x + self.residual(graph), self.skip(graph)


class GraphWaveNetForecast(nn.Module, _ProbabilisticOutputMixin):
    """Graph WaveNet-style dilated temporal and adaptive graph baseline."""

    def __init__(
        self,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        graph_prior: torch.Tensor,
        hidden: int = 64,
        layers: int = 6,
        dropout: float = 0.1,
        graph_rank: int = 10,
        **_: object,
    ) -> None:
        super().__init__()
        self.n_stations, self.n_pollutants, self.horizon = n_stations, n_pollutants, horizon
        self.register_buffer("graph_prior", graph_prior.float())
        self.node_source = nn.Parameter(torch.randn(n_stations, graph_rank) / math.sqrt(graph_rank))
        self.node_target = nn.Parameter(torch.randn(n_stations, graph_rank) / math.sqrt(graph_rank))
        self.input_projection = nn.Conv2d(n_features * 3 + 8, hidden, 1)
        self.blocks = nn.ModuleList(
            [_GraphWaveBlock(hidden, 2 ** (layer % 4), dropout) for layer in range(layers)]
        )
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU())
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def _adjacency(self) -> torch.Tensor:
        scores = torch.relu(self.node_source @ self.node_target.T) + self.graph_prior
        return torch.softmax(scores, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, _, n, _ = values.shape
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        sequence = torch.cat([values, mask, gaps, calendar], dim=-1)
        x = self.input_projection(sequence.permute(0, 3, 2, 1))
        adjacency = self._adjacency()
        skip = torch.zeros_like(x)
        for block in self.blocks:
            x, contribution = block(x, adjacency)
            skip = skip + contribution
        hidden = self.output(skip)[..., -1].permute(0, 2, 1)
        mu = self.head(hidden).reshape(b, n, self.horizon, self.output_width)
        mu = mu.permute(0, 2, 1, 3)
        output = self._wrap(mu)
        output["adjacency"] = adjacency[None].expand(b, -1, -1)
        return output
