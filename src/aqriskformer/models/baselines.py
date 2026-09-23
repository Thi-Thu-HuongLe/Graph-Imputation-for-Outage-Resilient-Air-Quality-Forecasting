from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import causal_mask


class _ProbabilisticOutputMixin:
    horizon: int
    n_stations: int
    n_pollutants: int
    log_scale: nn.Parameter

    @property
    def output_width(self) -> int:
        distribution = getattr(self, "output_distribution", "legacy_gaussian")
        parameters = {"legacy_gaussian": 1, "gaussian": 2, "student_t": 3}
        return self.n_pollutants * parameters[distribution]

    def configure_probabilistic_head(self, distribution: str) -> None:
        """Use backbone-dependent parameters at every station/horizon/pollutant.

        The default retains the archived checkpoint schema. Matched heads must
        be fitted from scratch; they cannot reuse constant-scale checkpoints.
        """
        if distribution not in {"legacy_gaussian", "gaussian", "student_t"}:
            raise ValueError(f"Unknown output distribution: {distribution}")
        if distribution == "legacy_gaussian":
            return
        self.output_distribution = distribution
        self.head = nn.Linear(self.head.in_features, self.horizon * self.output_width)
        del self.log_scale

    def _wrap(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        distribution = getattr(self, "output_distribution", "legacy_gaussian")
        if distribution == "legacy_gaussian":
            sigma = F.softplus(self.log_scale)[None, None, None, :].expand_as(mu) + 1e-4
            nu = torch.full_like(mu, 1e6)
        else:
            width = 3 if distribution == "student_t" else 2
            raw = mu.reshape(*mu.shape[:-1], self.n_pollutants, width)
            mu = raw[..., 0]
            sigma = F.softplus(raw[..., 1]) + 1e-4
            nu = (
                F.softplus(raw[..., 2]) + 2.01
                if distribution == "student_t"
                else torch.full_like(mu, 1e6)
            )
        return {
            # Keep likelihood and calibration arithmetic in float32 even when
            # the backbone runs under BF16 autocast.
            "mu": mu.float(),
            "sigma": sigma.float(),
            "nu": nu.float(),
            "adjacency": torch.eye(self.n_stations, device=mu.device)[None].expand(
                mu.shape[0], -1, -1
            ),
        }


class RecurrentForecast(nn.Module, _ProbabilisticOutputMixin):
    def __init__(
        self,
        cell: str,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon, self.n_stations, self.n_pollutants = horizon, n_stations, n_pollutants
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.encoder = recurrent(
            n_features * 3 + 8,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, length, n, _ = values.shape
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        x = torch.cat([values, mask, gaps, calendar], dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(b * n, length, -1)
        encoded, _ = self.encoder(x)
        mu = self.head(encoded[:, -1]).reshape(b, n, self.horizon, self.output_width)
        return self._wrap(mu.permute(0, 2, 1, 3))


class LSTMForecast(RecurrentForecast):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(cell="lstm", **kwargs)


class GRUForecast(RecurrentForecast):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(cell="gru", **kwargs)


class _CausalTCNBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.conv(x)
        if self.padding:
            output = output[..., : -self.padding]
        return residual + self.dropout(F.gelu(output))


class TCNForecast(nn.Module, _ProbabilisticOutputMixin):
    def __init__(
        self,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon, self.n_stations, self.n_pollutants = horizon, n_stations, n_pollutants
        input_dim = n_features * 3 + 8
        blocks: list[nn.Module] = [nn.Conv1d(input_dim, hidden, 1)]
        for layer in range(layers):
            blocks.append(_CausalTCNBlock(hidden, 2**layer, dropout))
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, length, n, _ = values.shape
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        x = torch.cat([values, mask, gaps, calendar], dim=-1)
        x = x.permute(0, 2, 3, 1).reshape(b * n, -1, length)
        encoded = self.network(x)
        mu = self.head(encoded[..., -1]).reshape(b, n, self.horizon, self.output_width)
        return self._wrap(mu.permute(0, 2, 1, 3))


class TemporalTransformer(nn.Module, _ProbabilisticOutputMixin):
    def __init__(
        self,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon, self.n_stations, self.n_pollutants = horizon, n_stations, n_pollutants
        self.projection = nn.Linear(n_features * 3 + 8, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, length, n, _ = values.shape
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        x = torch.cat([values, mask, gaps, calendar], dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(b * n, length, -1)
        x = self.projection(x)
        position = _sinusoidal_position(length, x.shape[-1], x.device)
        x = self.encoder(x + position[None], mask=causal_mask(length, x.device), is_causal=True)
        mu = self.head(x[:, -1]).reshape(b, n, self.horizon, self.output_width)
        return self._wrap(mu.permute(0, 2, 1, 3))


class PatchTSTForecast(nn.Module, _ProbabilisticOutputMixin):
    """Station-local patch Transformer baseline for multivariate forecasting."""

    def __init__(
        self,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        patch_length: int = 24,
        patch_stride: int = 12,
    ) -> None:
        super().__init__()
        self.horizon, self.n_stations, self.n_pollutants = horizon, n_stations, n_pollutants
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        input_dim = n_features * 3 + 8
        self.patch_projection = nn.Linear(input_dim * patch_length, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, length, n, _ = values.shape
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        x = torch.cat([values, mask, gaps, calendar], dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(b * n, length, -1)
        if length < self.patch_length:
            padding = self.patch_length - length
            x = F.pad(x.transpose(1, 2), (padding, 0), mode="replicate").transpose(1, 2)
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).flatten(start_dim=2)
        tokens = self.patch_projection(patches)
        tokens = (
            tokens + _sinusoidal_position(tokens.shape[1], tokens.shape[2], tokens.device)[None]
        )
        encoded = self.encoder(tokens)
        summary = self.norm(encoded.mean(dim=1))
        mu = self.head(summary).reshape(b, n, self.horizon, self.output_width)
        return self._wrap(mu.permute(0, 2, 1, 3))


class ITransformerForecast(nn.Module, _ProbabilisticOutputMixin):
    """Station-local inverted Transformer with variables represented as tokens."""

    def __init__(
        self,
        n_features: int,
        n_pollutants: int,
        n_stations: int,
        horizon: int,
        lookback: int,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon, self.n_stations, self.n_pollutants = horizon, n_stations, n_pollutants
        self.lookback = lookback
        self.input_dim = n_features * 3 + 8
        self.temporal_projection = nn.Linear(lookback, hidden)
        self.variable_embedding = nn.Parameter(torch.randn(self.input_dim, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, horizon * n_pollutants)
        self.log_scale = nn.Parameter(torch.zeros(n_pollutants))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        b, length, n, _ = values.shape
        if length != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, received {length}")
        calendar = batch["calendar"][:, :, None, :].expand(-1, -1, n, -1)
        x = torch.cat([values, mask, gaps, calendar], dim=-1)
        x = x.permute(0, 2, 3, 1).reshape(b * n, self.input_dim, length)
        tokens = self.temporal_projection(x) + self.variable_embedding[None]
        encoded = self.encoder(tokens)
        summary = self.norm(encoded.mean(dim=1))
        mu = self.head(summary).reshape(b, n, self.horizon, self.output_width)
        return self._wrap(mu.permute(0, 2, 1, 3))


def _sinusoidal_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    divisor = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
    return encoding
