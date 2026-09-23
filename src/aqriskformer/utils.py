from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(
    path: str | Path,
    value: Any,
    *,
    replace_attempts: int = 12,
    initial_retry_seconds: float = 0.05,
) -> None:
    """Atomically write JSON, tolerating transient Windows/OneDrive file locks."""
    if replace_attempts < 1:
        raise ValueError("replace_attempts must be at least 1")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        for attempt in range(replace_attempts):
            try:
                temporary.replace(target)
                return
            except PermissionError:
                if attempt + 1 == replace_attempts:
                    raise
                delay = min(initial_retry_seconds * (2**attempt), 1.0)
                time.sleep(delay)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def code_commit(project_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
    }
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_devices": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_properties(index).name,
                        "capability": ".".join(
                            str(part) for part in torch.cuda.get_device_capability(index)
                        ),
                        "total_memory_gib": round(
                            torch.cuda.get_device_properties(index).total_memory / 2**30, 2
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
                "bf16_supported": torch.cuda.is_bf16_supported()
                if torch.cuda.is_available()
                else False,
            }
        )
    except ImportError:
        snapshot["torch"] = None
    return snapshot
