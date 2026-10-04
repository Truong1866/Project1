from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    def __init__(self, data: dict | None = None, root: Path = ROOT_DIR):
        self._data: dict = data or {}
        self.root = Path(root)

    @classmethod
    def load(cls, root: Path = ROOT_DIR) -> "Config":
        data: dict = {}
        for name in ("config.yaml", "config.local.yaml"):
            path = Path(root) / name
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    _deep_merge(data, yaml.safe_load(f) or {})
        return cls(data, root)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        return copy.deepcopy(self._data.get(name, {}))

    def path(self, key: str, default: str | None = None) -> Path:
        """Lấy một đường dẫn trong config và quy về tuyệt đối (tính từ thư mục gốc project)."""
        value = self.get(key, default)
        p = Path(str(value))
        return p if p.is_absolute() else self.root / p