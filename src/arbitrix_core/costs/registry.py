from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

DEFAULT_COST_MODEL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "default": {
        "module": "arbitrix.core.costs.models.parameterized",
        "enabled": True,
        "source": "builtin",
    }
}


@dataclass
class CostModelConfigEntry:
    name: str
    module: str
    enabled: bool = True
    source: str = "config"
    description: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "module": self.module,
            "enabled": self.enabled,
            "source": self.source,
        }
        if self.description:
            data["description"] = self.description
        return data


def parse_cost_model_entry(name: str, raw: Dict[str, Any]) -> CostModelConfigEntry:
    data = copy.deepcopy(raw or {})
    defaults = DEFAULT_COST_MODEL_TEMPLATES.get(name, {})
    module = data.get("module") or defaults.get("module")
    if not module:
        raise ValueError(f"Cost model '{name}' is missing module definition")
    enabled = bool(data.get("enabled", defaults.get("enabled", True)))
    source = data.get("source") or defaults.get("source") or "config"
    description = data.get("description") or defaults.get("description")
    return CostModelConfigEntry(
        name=name,
        module=str(module),
        enabled=enabled,
        source=str(source),
        description=str(description) if description else None,
    )


def _cost_model_mapping(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    data = getattr(cfg, "cost_models", None)
    if isinstance(data, Mapping):
        return dict(data)
    return {}


def load_cost_model_entries(cfg: Any) -> Dict[str, CostModelConfigEntry]:
    entries: Dict[str, CostModelConfigEntry] = {}
    raw_mapping = _cost_model_mapping(cfg)
    for name, raw in raw_mapping.items():
        try:
            entry = parse_cost_model_entry(name, raw)
        except ValueError:
            continue
        entries[name] = entry
    if "default" not in entries:
        entries["default"] = parse_cost_model_entry("default", {})
    return entries


def resolve_cost_model_module(cfg: Any, identifier: Optional[str]) -> Optional[str]:
    if not identifier:
        return DEFAULT_COST_MODEL_TEMPLATES["default"]["module"]
    normalized = identifier.strip()
    entries = load_cost_model_entries(cfg)
    entry = entries.get(normalized)
    if entry is not None:
        return entry.module
    if "." in normalized:
        return normalized
    return normalized


__all__ = [
    "CostModelConfigEntry",
    "DEFAULT_COST_MODEL_TEMPLATES",
    "load_cost_model_entries",
    "resolve_cost_model_module",
]
