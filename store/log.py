"""Append-only JSONL event log + content-addressing for configs.

Canonical JSON (sorted keys, no whitespace, ASCII) makes both the log lines and
the config hash stable across machines and Python versions — a prerequisite for
"same seed -> byte-identical log" and for verifiable benchmark suite hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from engine.events import Event


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def config_hash(rules: dict, seed: int) -> str:
    return hashlib.sha256(
        canonical_json({"rules": rules, "seed": seed}).encode("ascii")
    ).hexdigest()


class EventLogWriter:
    """Appends one canonical-JSON event per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="ascii")

    def write(self, event: Event) -> None:
        self._fh.write(canonical_json(event.model_dump(mode="json")) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_events(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="ascii") as fh:
        return [json.loads(line) for line in fh if line.strip()]
