from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from aqriskformer.outage_models import build_outage_model
from aqriskformer.utils import sha256_file
from scripts.run_journal_development import (
    initialize_impute_adapter,
    load_training_checkpoint,
    preserve_legacy_partial_run,
    save_training_checkpoint,
    scientifically_compatible_runner_upgrade,
)


def test_epoch_checkpoint_restores_optimizer_and_rng_state(tmp_path) -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loader = DataLoader(
        TensorDataset(torch.arange(10)),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(17),
    )
    list(loader)
    loss = model(torch.ones(3, 2)).square().mean()
    loss.backward()
    optimizer.step()
    protocol = {"name": "test"}
    training = {"epochs": 3, "patience": 2}
    history = [{"epoch": 1, "selection_score": 0.4}]
    checkpoint_path = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        loader,
        protocol,
        training,
        "tiny",
        42,
        1,
        0.4,
        1,
        0,
        history,
        1.5,
        None,
        torch.device("cpu"),
    )
    expected_python = random.random()
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)
    expected_order = torch.cat([batch[0] for batch in loader])

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_loader = DataLoader(
        TensorDataset(torch.arange(10)),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(99),
    )
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_loader,
        protocol,
        training,
        "tiny",
        42,
        torch.device("cpu"),
    )

    assert checkpoint is not None
    assert checkpoint["epoch"] == 1
    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)
    torch.testing.assert_close(torch.rand(3), expected_torch)
    torch.testing.assert_close(
        torch.cat([batch[0] for batch in restored_loader]), expected_order
    )
    for actual, expected in zip(restored_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_legacy_partial_run_is_preserved_before_restart(tmp_path) -> None:
    run = tmp_path / "dcrnn" / "seed_42"
    run.mkdir(parents=True)
    torch.save({"epoch": 7, "model": {}}, run / "best.pt")
    (run / "history.json").write_text('[{"epoch": 7}]', encoding="utf-8")

    snapshot = preserve_legacy_partial_run(run)

    assert snapshot == run / "resume_snapshots" / "legacy_partial_epoch_007"
    assert (snapshot / "best.pt").is_file()
    assert (snapshot / "history.json").is_file()
    assert (snapshot / "snapshot.json").is_file()


def test_runner_only_signature_change_is_scientifically_compatible() -> None:
    runner = "scripts/run_journal_development.py"
    previous = {
        "protocol": {"epochs": 20},
        "prepared_sha256": "data",
        "source_hashes": {runner: "old-runner", "src/model.py": "same-model"},
        "digest": "old",
    }
    current = {
        "protocol": {"epochs": 20},
        "prepared_sha256": "data",
        "source_hashes": {runner: "new-runner", "src/model.py": "same-model"},
        "digest": "new",
    }

    assert scientifically_compatible_runner_upgrade(previous, current)
    current["source_hashes"]["src/model.py"] = "changed-model"
    assert not scientifically_compatible_runner_upgrade(previous, current)


def test_impute_adapter_initialization_loads_and_freezes_shared_path(tmp_path) -> None:
    graph = torch.ones(3, 3) - torch.eye(3)
    common = {
        "features": 5,
        "pollutants": 5,
        "stations": 3,
        "horizon": 4,
        "graph": graph,
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
    }
    protocol = {
        "prepared_development": "development.npz",
        "lookback": 24,
        "horizon": 4,
        "reported_horizons": [1, 4],
        "architecture": {"hidden": 8, "layers": 1},
        "training": {"epochs": 3},
        "fill_limit": 6,
    }
    torch.manual_seed(1)
    base = build_outage_model("impute_then_local_tcn", **common)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model": base.state_dict(),
            "protocol": protocol,
            "model_name": "impute_then_local_tcn",
            "seed": 42,
        },
        checkpoint_path,
    )
    torch.manual_seed(2)
    adapter = build_outage_model("impute_spatial_adapter", **common)

    digest = initialize_impute_adapter(
        adapter, checkpoint_path, protocol, 42, torch.device("cpu")
    )

    assert digest == sha256_file(checkpoint_path)
    for key, value in base.state_dict().items():
        if key == "graph" or key.startswith(("encoder.", "local_head.")):
            torch.testing.assert_close(adapter.state_dict()[key], value)
    assert all(not parameter.requires_grad for parameter in adapter.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in adapter.local_head.parameters())
    assert any(parameter.requires_grad for parameter in adapter.residual.parameters())
