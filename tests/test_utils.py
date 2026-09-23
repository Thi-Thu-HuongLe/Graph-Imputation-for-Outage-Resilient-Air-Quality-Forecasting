from __future__ import annotations

from pathlib import Path

import pytest

from aqriskformer import utils


def test_write_json_retries_transient_replace_lock(tmp_path, monkeypatch) -> None:
    target = tmp_path / "status.json"
    original_replace = Path.replace
    calls = 0

    def flaky_replace(source: Path, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(13, "transient OneDrive lock")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)

    utils.write_json(target, {"state": "complete"}, replace_attempts=3)

    assert utils.read_json(target) == {"state": "complete"}
    assert calls == 3
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_rejects_empty_retry_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        utils.write_json(tmp_path / "status.json", {}, replace_attempts=0)
