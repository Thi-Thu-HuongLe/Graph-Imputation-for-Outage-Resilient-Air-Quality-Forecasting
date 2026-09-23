from __future__ import annotations

import numpy as np
import torch

from aqriskformer.epa_aqs import coordinate_knn_graph, haversine_distances
from aqriskformer.outage_data import (
    OutageData,
    OutageWindows,
    causal_outage_inputs,
    natural_comissingness_origins,
)
from aqriskformer.outage_evaluation import (
    apply_sigma_calibration,
    fit_multiplicative_sigma_calibration,
    matched_gaussian_loss,
    scaled_thresholds,
)
from aqriskformer.outage_models import LEARNED_MODELS, build_outage_model


def synthetic_data(length: int = 300) -> OutageData:
    rng = np.random.default_rng(2)
    stations, pollutants = 3, 5
    return OutageData(
        raw=rng.normal(size=(length, stations, pollutants)).astype(np.float32),
        observed=np.ones((length, stations, pollutants), dtype=bool),
        calendar=np.zeros((length, 8), dtype=np.float32),
        center=np.zeros((stations, pollutants), dtype=np.float32),
        scale=np.ones((stations, pollutants), dtype=np.float32),
        thresholds=np.tile(np.array([[0.5, 1.0, 1.5]], dtype=np.float32), (pollutants, 1)),
        mase=np.ones((stations, pollutants), dtype=np.float32),
        timestamps=np.arange(length).astype("datetime64[h]"),
        stations=["a", "b", "c"],
        pollutants=["PM2.5", "NO2", "O3", "CO", "SO2"],
        graph=np.ones((stations, stations), dtype=np.float32) - np.eye(stations, dtype=np.float32),
        split_bounds={"train": (0, 199), "validation": (220, 299)},
    )


def test_outage_windows_use_dynamic_pollutant_count_and_embargo() -> None:
    data = synthetic_data()
    protocol = {"lookback": 24, "horizon": 6, "train_stride": 1, "validation_stride": 1}
    train = OutageWindows(data, protocol, "train")
    validation = OutageWindows(data, protocol, "validation")
    assert train.origins[[0, -1]].tolist() == [23, 193]
    assert validation.origins[[0, -1]].tolist() == [219, 293]
    assert validation[0]["target"].shape == (6, 3, 5)


def test_trailing_outage_is_target_station_specific_and_causal() -> None:
    data = synthetic_data(length=40)
    batch = {
        "raw": torch.from_numpy(data.raw[:24][None]),
        "mask": torch.from_numpy(data.observed[:24][None]),
        "calendar": torch.from_numpy(data.calendar[:24][None]),
        "target": torch.zeros(1, 2, 3, 5),
        "target_mask": torch.ones(1, 2, 3, 5, dtype=torch.bool),
        "origin": torch.tensor([23]),
    }
    output = causal_outage_inputs(
        batch,
        "station_trailing_6h_1",
        torch.Generator().manual_seed(1),
        fill_limit=6,
    )
    assert not output["mask"][:, -6:, 1].any()
    assert output["mask"][:, :-6, 1].all()
    assert output["mask"][:, :, 0].all() and output["mask"][:, :, 2].all()
    assert torch.all(output["gaps"][:, -1, 1] == torch.log1p(torch.tensor(6.0)))


def test_all_learned_models_share_gaussian_output_shape() -> None:
    data = synthetic_data(length=40)
    batch = {
        "values": torch.from_numpy(data.raw[:24][None]),
        "mask": torch.from_numpy(data.observed[:24][None]),
        "gaps": torch.zeros(1, 24, 3, 5),
        "calendar": torch.from_numpy(data.calendar[:24][None]),
    }
    for name in LEARNED_MODELS:
        torch.manual_seed(3)
        model = build_outage_model(
            name,
            features=5,
            pollutants=5,
            stations=3,
            horizon=4,
            graph=torch.from_numpy(data.graph),
            hidden=8,
            layers=1,
            dropout=0.0,
        )
        output = model(batch)
        assert output["mu"].shape == output["sigma"].shape == (1, 4, 3, 5)
        assert torch.isfinite(output["mu"]).all()
        assert torch.all(output["sigma"] > 0)


def test_outage_gated_residual_is_zero_for_fully_observed_origin() -> None:
    data = synthetic_data(length=40)
    batch = {
        "values": torch.from_numpy(data.raw[:24][None]),
        "mask": torch.from_numpy(data.observed[:24][None]),
        "gaps": torch.zeros(1, 24, 3, 5),
        "calendar": torch.from_numpy(data.calendar[:24][None]),
    }
    model = build_outage_model(
        "outage_gated_spatial_residual",
        features=5,
        pollutants=5,
        stations=3,
        horizon=4,
        graph=torch.from_numpy(data.graph),
        hidden=8,
        layers=1,
        dropout=0.0,
    )
    model.eval()

    with torch.inference_mode():
        output = model(batch)
        state = model.encode(model.imputed_batch(batch))
        local_only = model.gaussian_output(model.local_parameters(state))

    torch.testing.assert_close(model.outage_gate(batch), torch.zeros(1, 3))
    torch.testing.assert_close(output["mu"], local_only["mu"])
    torch.testing.assert_close(output["sigma"], local_only["sigma"])


def test_outage_gate_reaches_one_after_six_missing_hours() -> None:
    data = synthetic_data(length=40)
    batch = {
        "values": torch.from_numpy(data.raw[:24][None]),
        "mask": torch.from_numpy(data.observed[:24][None]),
        "gaps": torch.zeros(1, 24, 3, 5),
        "calendar": torch.from_numpy(data.calendar[:24][None]),
    }
    batch["mask"][:, -6:, 1] = False
    batch["gaps"][:, -1, 1] = torch.log1p(torch.tensor(6.0))
    model = build_outage_model(
        "outage_gated_spatial_residual",
        features=5,
        pollutants=5,
        stations=3,
        horizon=4,
        graph=torch.from_numpy(data.graph),
        hidden=8,
        layers=1,
        dropout=0.0,
    )

    expected = torch.tensor([[0.0, 1.0, 0.0]])
    torch.testing.assert_close(model.outage_gate(batch), expected)


def test_adapter_gate_scales_with_outage_duration() -> None:
    data = synthetic_data(length=40)
    batch = {
        "values": torch.from_numpy(data.raw[:24][None]),
        "mask": torch.from_numpy(data.observed[:24][None]),
        "gaps": torch.zeros(1, 24, 3, 5),
        "calendar": torch.from_numpy(data.calendar[:24][None]),
    }
    batch["mask"][:, -6:, 1] = False
    batch["gaps"][:, -1, 1] = torch.log1p(torch.tensor(6.0))
    model = build_outage_model(
        "impute_spatial_adapter",
        features=5,
        pollutants=5,
        stations=3,
        horizon=4,
        graph=torch.from_numpy(data.graph),
        hidden=8,
        layers=1,
        dropout=0.0,
    )

    expected_station_gate = torch.log1p(torch.tensor(6.0)) / torch.log1p(torch.tensor(24.0))
    expected = torch.tensor([[0.0, expected_station_gate, 0.0]])
    torch.testing.assert_close(model.outage_gate(batch), expected)


def test_gated_candidate_preserves_spatial_residual_parameter_budget() -> None:
    data = synthetic_data(length=40)
    common = {
        "features": 5,
        "pollutants": 5,
        "stations": 3,
        "horizon": 4,
        "graph": torch.from_numpy(data.graph),
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
    }
    spatial = build_outage_model("spatial_residual", **common)
    gated = build_outage_model("outage_gated_spatial_residual", **common)

    assert sum(p.numel() for p in gated.parameters()) == sum(
        p.numel() for p in spatial.parameters()
    )


def test_adaptive_graph_starts_from_fixed_imputation_graph() -> None:
    data = synthetic_data(length=40)
    model = build_outage_model(
        "adaptive_graph_impute_tcn",
        features=5,
        pollutants=5,
        stations=3,
        horizon=4,
        graph=torch.from_numpy(data.graph),
        hidden=8,
        layers=1,
        dropout=0.0,
    )

    expected = torch.from_numpy(data.graph)[None].expand(5, -1, -1)
    torch.testing.assert_close(model.adaptive_graph(), expected)
    model.freeze_base()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.local_head.parameters())
    assert model.edge_logits.requires_grad


def test_matched_loss_accepts_five_pollutants() -> None:
    data = synthetic_data(length=40)
    target = torch.randn(2, 4, 3, 5)
    output = {"mu": torch.zeros_like(target), "sigma": torch.ones_like(target)}
    threshold = torch.from_numpy(scaled_thresholds(data))
    loss = matched_gaussian_loss(
        output,
        target,
        torch.ones_like(target, dtype=torch.bool),
        threshold,
        {"nll": 1.0, "huber": 0.5, "brier": 2.0, "quantile_weights": [1, 2, 4]},
    )
    assert torch.isfinite(loss) and loss > 0


def test_coordinate_graph_is_symmetric_and_geographic() -> None:
    coordinates = np.array(
        [[40.0, -112.0], [40.1, -112.0], [40.3, -112.0], [40.7, -112.0]],
        dtype=np.float32,
    )
    graph, distances = coordinate_knn_graph(coordinates, neighbors=1)
    np.testing.assert_allclose(graph, graph.T)
    np.testing.assert_allclose(np.diag(graph), 0)
    assert np.all((graph > 0).sum(axis=1) >= 1)
    np.testing.assert_allclose(distances, haversine_distances(coordinates))
    assert distances[0, 1] < distances[0, 2] < distances[0, 3]


def test_natural_comissingness_requires_all_core_channels_and_a_neighbor() -> None:
    data = synthetic_data(length=40)
    data.observed[10:16, 1, :3] = False
    data.observed[12:16, 1, 0] = True
    selected = natural_comissingness_origins(data, np.array([15, 16]), minimum_hours=6)
    assert not selected.any()
    data.observed[12:16, 1, 0] = False
    selected = natural_comissingness_origins(data, np.array([15, 16]), minimum_hours=6)
    assert selected[0, 1]
    assert not selected[1, 1]
    data.observed[15, 0, :3] = False
    data.observed[15, 2, :3] = False
    selected = natural_comissingness_origins(data, np.array([15]), minimum_hours=6)
    assert not selected[0, 1]


def test_closed_form_sigma_calibration_is_pollutant_specific_and_bounded() -> None:
    target = np.zeros((2, 1, 1, 2), dtype=np.float32)
    target[..., 0] = 2.0
    target[..., 1] = 10.0
    prediction = {
        "mu": np.zeros_like(target),
        "sigma": np.ones_like(target),
        "target": target,
        "mask": np.ones_like(target, dtype=bool),
        "origin": np.arange(2),
    }
    factors = fit_multiplicative_sigma_calibration(prediction, (0.25, 4.0))
    np.testing.assert_allclose(factors, [2.0, 4.0])
    calibrated = apply_sigma_calibration(prediction, factors)
    np.testing.assert_allclose(calibrated["sigma"][..., 0], 2.0)
    np.testing.assert_allclose(calibrated["sigma"][..., 1], 4.0)
