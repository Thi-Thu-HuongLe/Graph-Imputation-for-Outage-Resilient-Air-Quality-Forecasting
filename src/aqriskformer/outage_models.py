"""Models for the sensor-outage-resilience comparison protocol."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .models.baselines import _CausalTCNBlock
from .models.graph_baselines import DCRNNForecast, GraphWaveNetForecast

LEARNED_MODELS = (
    "local_tcn",
    "local_capacity_residual",
    "spatial_residual",
    "impute_then_local_tcn",
    "outage_gated_spatial_residual",
    "impute_spatial_adapter",
    "adaptive_graph_impute_tcn",
    "dcrnn",
    "graph_wavenet",
)
DETERMINISTIC_MODELS = ("persistence", "seasonal_naive_24h")


class LocalGaussianTCN(nn.Module):
    """Station-local causal TCN with horizon-dependent Gaussian parameters."""

    def __init__(
        self,
        features: int,
        pollutants: int,
        stations: int,
        horizon: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = features
        self.pollutants = pollutants
        self.stations = stations
        self.horizon = horizon
        self.encoder = nn.Sequential(
            nn.Conv1d(features * 3 + 8, hidden, 1),
            *[_CausalTCNBlock(hidden, 2**layer, dropout) for layer in range(layers)],
        )
        self.local_head = nn.Linear(hidden, horizon * pollutants * 2)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values, mask, gaps = batch["values"], batch["mask"], batch["gaps"]
        batch_size, length, stations, _ = values.shape
        if stations != self.stations:
            raise ValueError(f"Expected {self.stations} stations, received {stations}")
        calendar = batch["calendar"][:, :, None].expand(-1, -1, stations, -1)
        inputs = torch.cat((values, mask.float(), gaps, calendar), dim=-1)
        encoded = self.encoder(
            inputs.permute(0, 2, 3, 1).reshape(batch_size * stations, -1, length)
        )
        return encoded[:, :, -1].reshape(batch_size, stations, -1)

    def local_parameters(self, state: torch.Tensor) -> torch.Tensor:
        batch_size, stations, _ = state.shape
        return (
            self.local_head(state)
            .reshape(batch_size, stations, self.horizon, self.pollutants, 2)
            .permute(0, 2, 1, 3, 4)
        )

    @staticmethod
    def gaussian_output(raw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mu": raw[..., 0].float(),
            "sigma": F.softplus(raw[..., 1].float()) + 1e-4,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.gaussian_output(self.local_parameters(self.encode(batch)))


class SpatialResidualForecast(LocalGaussianTCN):
    """Add an always-on learned cross-station residual to the local TCN path."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        hidden = self.local_head.in_features
        self.query = nn.Linear(hidden, 32)
        self.key = nn.Linear(hidden, 32)
        self.value = nn.Linear(hidden, 32)
        self.local_projection = nn.Linear(hidden, 32)
        self.pollutant_embedding = nn.Embedding(self.pollutants, 8)
        positions = torch.arange(1, self.horizon + 1).float()[:, None]
        frequency = torch.exp(torch.arange(0, 16, 2).float() * (-math.log(10000) / 16))
        encoding = torch.zeros(self.horizon, 16)
        encoding[:, 0::2] = (positions * frequency).sin()
        encoding[:, 1::2] = (positions * frequency).cos()
        self.register_buffer("horizon_encoding", encoding)
        self.register_buffer("self_mask", torch.eye(self.stations, dtype=torch.bool))
        self.residual = nn.Sequential(nn.Linear(88, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def residual_features(self, state: torch.Tensor, local_only: bool = False) -> torch.Tensor:
        batch_size, stations, _ = state.shape
        if local_only:
            local_weight = torch.sigmoid(
                (self.query(state) * self.key(state)).sum(-1, keepdim=True).float() / math.sqrt(32)
            ).to(state.dtype)
            neighbor = local_weight * self.value(state)
        else:
            attention = (self.query(state) @ self.key(state).transpose(-1, -2)) / math.sqrt(32)
            attention = attention.float().masked_fill(self.self_mask, -torch.inf).softmax(-1)
            neighbor = attention.to(state.dtype) @ self.value(state)
        shape = (batch_size, self.horizon, stations, self.pollutants, -1)
        return torch.cat(
            (
                self.local_projection(state)[:, None, :, None].expand(shape),
                neighbor[:, None, :, None].expand(shape),
                self.horizon_encoding[None, :, None, None].expand(shape),
                self.pollutant_embedding.weight[None, None, None].expand(shape),
            ),
            dim=-1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = self.encode(batch)
        raw = self.local_parameters(state) + self.residual(self.residual_features(state))
        return self.gaussian_output(raw)


class LocalCapacityResidualForecast(SpatialResidualForecast):
    """Parameter-matched control whose residual cannot access another station."""

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = self.encode(batch)
        raw = self.local_parameters(state) + self.residual(
            self.residual_features(state, local_only=True)
        )
        return self.gaussian_output(raw)


class ImputeThenLocalTCN(LocalGaussianTCN):
    """Coordinate-graph interpolation followed by the same local TCN backbone."""

    def __init__(self, *args: object, graph: torch.Tensor, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if graph.shape != (self.stations, self.stations):
            raise ValueError("Graph shape differs from the station count")
        self.register_buffer("graph", graph.float())

    def imputed_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"].bool()
        values = batch["values"]
        available = mask.to(values.dtype)
        numerator = torch.einsum("ij,btjf->btif", self.graph, values * available)
        denominator = torch.einsum("ij,btjf->btif", self.graph, available)
        neighbor = numerator / denominator.clamp_min(1e-8)
        imputed = torch.where(mask, values, torch.where(denominator > 0, neighbor, values))
        complete = {
            **batch,
            "values": imputed,
            "mask": torch.ones_like(mask),
            "gaps": torch.zeros_like(batch["gaps"]),
        }
        return complete

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return super().forward(self.imputed_batch(batch))


class OutageGatedSpatialResidualForecast(SpatialResidualForecast):
    """Graph-imputed local forecast plus a residual activated by trailing outages.

    The local path is identical to ``ImputeThenLocalTCN``. The cross-station
    residual is exactly zero when the forecast-origin channels are observed and
    reaches full strength after six or more hours of trailing missingness. This
    protects clean-input performance while retaining a learned spatial correction
    for the prespecified 6 h and 24 h station-outage conditions.
    """

    def __init__(self, *args: object, graph: torch.Tensor, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if graph.shape != (self.stations, self.stations):
            raise ValueError("Graph shape differs from the station count")
        self.register_buffer("graph", graph.float())

    def imputed_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"].bool()
        values = batch["values"]
        available = mask.to(values.dtype)
        numerator = torch.einsum("ij,btjf->btif", self.graph, values * available)
        denominator = torch.einsum("ij,btjf->btif", self.graph, available)
        neighbor = numerator / denominator.clamp_min(1e-8)
        imputed = torch.where(mask, values, torch.where(denominator > 0, neighbor, values))
        return {
            **batch,
            "values": imputed,
            "mask": torch.ones_like(mask),
            "gaps": torch.zeros_like(batch["gaps"]),
        }

    @staticmethod
    def outage_gate(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"][:, -1].bool()
        age_hours = torch.expm1(batch["gaps"][:, -1].float()).clamp_min(0)
        channel_gate = (~mask).float() * (age_hours / 6.0).clamp(max=1.0)
        return channel_gate.mean(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = self.encode(self.imputed_batch(batch))
        local = self.local_parameters(state)
        spatial = self.residual(self.residual_features(state))
        gate = self.outage_gate(batch)[:, None, :, None, None].to(spatial.dtype)
        return self.gaussian_output(local + gate * spatial)


class ImputeSpatialAdapterForecast(OutageGatedSpatialResidualForecast):
    """Mean-only spatial adapter for a frozen graph-imputed local forecaster."""

    @staticmethod
    def outage_gate(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"][:, -1].bool()
        normalized_age = batch["gaps"][:, -1].float() / math.log1p(batch["gaps"].shape[1])
        return ((~mask).float() * normalized_age.clamp(0.0, 1.0)).mean(-1)

    def freeze_base(self) -> None:
        for module in (self.encoder, self.local_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = self.encode(self.imputed_batch(batch))
        local = self.local_parameters(state)
        spatial_mean = self.residual(self.residual_features(state))[..., 0]
        gate = self.outage_gate(batch)[:, None, :, None].to(spatial_mean.dtype)
        raw = torch.stack((local[..., 0] + gate * spatial_mean, local[..., 1]), dim=-1)
        return self.gaussian_output(raw)


class AdaptiveGraphImputeTCN(ImputeThenLocalTCN):
    """Pollutant-specific positive graph weights over a frozen local forecaster."""

    def __init__(self, *args: object, graph: torch.Tensor, **kwargs: object) -> None:
        super().__init__(*args, graph=graph, **kwargs)
        positive = graph.float().clamp_min(1e-6)
        inverse_softplus = torch.log(torch.expm1(positive))
        self.edge_logits = nn.Parameter(
            inverse_softplus[None].expand(self.features, -1, -1).clone()
        )
        self.register_buffer("edge_mask", (graph > 0)[None])

    def adaptive_graph(self) -> torch.Tensor:
        return F.softplus(self.edge_logits) * self.edge_mask

    def imputed_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"].bool()
        values = batch["values"]
        available = mask.to(values.dtype)
        weights = self.adaptive_graph().to(values.dtype)
        numerator = torch.einsum("fij,btjf->btif", weights, values * available)
        denominator = torch.einsum("fij,btjf->btif", weights, available)
        neighbor = numerator / denominator.clamp_min(1e-8)
        imputed = torch.where(mask, values, torch.where(denominator > 0, neighbor, values))
        return {
            **batch,
            "values": imputed,
            "mask": torch.ones_like(mask),
            "gaps": torch.zeros_like(batch["gaps"]),
        }

    def freeze_base(self) -> None:
        for module in (self.encoder, self.local_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def build_outage_model(
    name: str,
    *,
    features: int,
    pollutants: int,
    stations: int,
    horizon: int,
    graph: torch.Tensor,
    hidden: int,
    layers: int,
    dropout: float,
) -> nn.Module:
    common = {
        "features": features,
        "pollutants": pollutants,
        "stations": stations,
        "horizon": horizon,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
    }
    if name == "local_tcn":
        return LocalGaussianTCN(**common)
    if name == "local_capacity_residual":
        return LocalCapacityResidualForecast(**common)
    if name == "spatial_residual":
        return SpatialResidualForecast(**common)
    if name == "impute_then_local_tcn":
        return ImputeThenLocalTCN(**common, graph=graph)
    if name == "outage_gated_spatial_residual":
        return OutageGatedSpatialResidualForecast(**common, graph=graph)
    if name == "impute_spatial_adapter":
        return ImputeSpatialAdapterForecast(**common, graph=graph)
    if name == "adaptive_graph_impute_tcn":
        return AdaptiveGraphImputeTCN(**common, graph=graph)
    graph_common = {
        "n_features": features,
        "n_pollutants": pollutants,
        "n_stations": stations,
        "horizon": horizon,
        "graph_prior": graph,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
    }
    if name == "dcrnn":
        model = DCRNNForecast(**graph_common)
    elif name == "graph_wavenet":
        model = GraphWaveNetForecast(**graph_common)
    else:
        raise KeyError(f"Unknown learned outage model: {name}")
    model.configure_probabilistic_head("gaussian")
    return model


def _latest_observation(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    raw, mask = batch["raw"], batch["mask"].bool()
    length = raw.shape[1]
    time = torch.arange(length, device=raw.device)[None, :, None, None]
    last = torch.where(mask, time, -1).cummax(1).values[:, -1]
    gathered = raw.gather(1, last.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(last >= 0, gathered, torch.zeros_like(gathered))


def deterministic_forecast(
    name: str,
    batch: dict[str, torch.Tensor],
    sigma: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Forecast after outages using only observations still available in the input."""
    if name not in DETERMINISTIC_MODELS:
        raise KeyError(f"Unknown deterministic model: {name}")
    raw, mask = batch["raw"], batch["mask"].bool()
    batch_size, length, stations, pollutants = raw.shape
    horizon = sigma.shape[0]
    latest = _latest_observation(batch)
    if name == "persistence":
        mu = latest[:, None].expand(-1, horizon, -1, -1)
    else:
        forecasts = []
        for step in range(1, horizon + 1):
            value = latest.clone()
            found = torch.zeros(
                (batch_size, stations, pollutants), device=raw.device, dtype=torch.bool
            )
            cycle = math.ceil(step / 24)
            index = length - 1 + step - 24 * cycle
            while index >= 0:
                available = mask[:, index] & ~found
                value = torch.where(available, raw[:, index], value)
                found |= available
                index -= 24
            forecasts.append(value)
        mu = torch.stack(forecasts, dim=1)
    return {
        "mu": mu.float(),
        "sigma": sigma[None].expand(batch_size, -1, -1, -1).float(),
    }
