from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

SPREAD_MODES = {"static", "provider_only", "stochastic_only", "provider_plus_stochastic"}
DEFAULT_DISTRIBUTION = "lognormal"


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class SpreadRuntimeConfig:
    mode: str
    explicit_mode: bool
    static_points: Optional[float]
    provider_fallback_points: float
    stochastic_enabled: bool
    distribution: str
    mu: float
    sigma: float
    normal_mean: float
    normal_std: float
    min_points: float
    max_points: Optional[float]
    volatility_multiplier: float
    size_multiplier: float
    reference_size_lot: float
    hourly_multipliers: Dict[int, float]
    weekday_multipliers: Dict[int, float]


def _extract_config(model_parameters: Optional[Dict[str, Any]]) -> Optional[SpreadRuntimeConfig]:
    params = _as_dict(model_parameters)
    spread_model = _as_dict(params.get("spread_model"))
    stochastic = _as_dict(spread_model.get("stochastic"))

    explicit_mode = "mode" in spread_model or "spread_mode" in params
    mode_raw = spread_model.get("mode", params.get("spread_mode"))
    mode = str(mode_raw or "static").strip().lower()
    if mode not in SPREAD_MODES:
        mode = "static"

    static_present = "static_points" in spread_model or "spread_points" in params
    static_points = _as_float(
        spread_model.get("static_points", params.get("spread_points")),
        default=None,
    )

    stochastic_has_inputs = bool(stochastic) or any(
        key in params
        for key in (
            "spread_stochastic_distribution",
            "spread_stochastic_mu",
            "spread_stochastic_sigma",
            "spread_stochastic_min_points",
            "spread_stochastic_max_points",
            "spread_stochastic_volatility_multiplier",
            "spread_stochastic_size_multiplier",
        )
    )
    stochastic_enabled = bool(
        stochastic.get("enabled", params.get("spread_stochastic_enabled", stochastic_has_inputs))
    )

    # Keep legacy behaviour when no explicit spread-mode settings are provided.
    if not explicit_mode and not stochastic_has_inputs:
        return None

    distribution = str(
        stochastic.get("distribution", params.get("spread_stochastic_distribution", DEFAULT_DISTRIBUTION))
    ).strip().lower()
    if distribution not in {"lognormal", "normal"}:
        distribution = DEFAULT_DISTRIBUTION

    static_for_defaults = (
        static_points
        if static_points is not None and static_points > 0
        else _as_float(params.get("spread_points"), default=0.1) or 0.1
    )
    mu_default = math.log(max(static_for_defaults, 1e-6))
    sigma_default = 0.35
    normal_std_default = max(static_for_defaults * 0.25, 1e-6)

    mu = _as_float(stochastic.get("mu", params.get("spread_stochastic_mu")), default=mu_default) or mu_default
    sigma = max(
        _as_float(stochastic.get("sigma", params.get("spread_stochastic_sigma")), default=sigma_default)
        or sigma_default,
        0.0,
    )
    normal_mean = _as_float(
        stochastic.get("mean", params.get("spread_stochastic_mean")),
        default=static_for_defaults,
    ) or static_for_defaults
    normal_std = max(
        _as_float(stochastic.get("std", params.get("spread_stochastic_std")), default=normal_std_default)
        or normal_std_default,
        0.0,
    )
    min_points = max(
        _as_float(
            stochastic.get("min_points", params.get("spread_stochastic_min_points")),
            default=0.0,
        )
        or 0.0,
        0.0,
    )
    max_points = _as_float(
        stochastic.get("max_points", params.get("spread_stochastic_max_points")),
        default=None,
    )
    if max_points is not None and max_points < min_points:
        max_points = min_points

    provider_fallback_points = max(
        _as_float(
            spread_model.get(
                "provider_fallback_points",
                params.get("spread_provider_fallback_points", 0.0),
            ),
            default=0.0,
        )
        or 0.0,
        0.0,
    )

    volatility_multiplier = (
        _as_float(
            stochastic.get(
                "volatility_multiplier",
                params.get("spread_stochastic_volatility_multiplier", 0.0),
            ),
            default=0.0,
        )
        or 0.0
    )
    size_multiplier = (
        _as_float(
            stochastic.get(
                "size_multiplier",
                params.get("spread_stochastic_size_multiplier", 0.0),
            ),
            default=0.0,
        )
        or 0.0
    )
    reference_size_lot = max(
        _as_float(
            stochastic.get(
                "reference_size_lot",
                params.get("spread_stochastic_reference_size_lot", 1.0),
            ),
            default=1.0,
        )
        or 1.0,
        1e-6,
    )

    hourly_raw = _as_dict(stochastic.get("hourly_multipliers"))
    weekday_raw = _as_dict(stochastic.get("weekday_multipliers"))

    def _normalize_int_key_map(raw: Dict[str, Any]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for key, value in raw.items():
            try:
                parsed_key = int(key)
            except (TypeError, ValueError):
                continue
            parsed_value = _as_float(value, default=None)
            if parsed_value is None or parsed_value <= 0:
                continue
            out[parsed_key] = float(parsed_value)
        return out

    return SpreadRuntimeConfig(
        mode=mode,
        explicit_mode=explicit_mode,
        static_points=static_points if static_present else None,
        provider_fallback_points=provider_fallback_points,
        stochastic_enabled=stochastic_enabled,
        distribution=distribution,
        mu=mu,
        sigma=sigma,
        normal_mean=normal_mean,
        normal_std=normal_std,
        min_points=min_points,
        max_points=max_points,
        volatility_multiplier=volatility_multiplier,
        size_multiplier=size_multiplier,
        reference_size_lot=reference_size_lot,
        hourly_multipliers=_normalize_int_key_map(hourly_raw),
        weekday_multipliers=_normalize_int_key_map(weekday_raw),
    )


def _stochastic_spread(
    frame: pd.DataFrame,
    cfg: SpreadRuntimeConfig,
    *,
    seed: Optional[int] = None,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    rng = np.random.default_rng(seed)
    size = len(frame)
    if cfg.distribution == "normal":
        samples = rng.normal(loc=cfg.normal_mean, scale=cfg.normal_std, size=size)
    else:
        samples = rng.lognormal(mean=cfg.mu, sigma=cfg.sigma, size=size)

    spread = np.maximum(samples, 0.0)
    if cfg.min_points > 0:
        spread = np.maximum(spread, cfg.min_points)
    if cfg.max_points is not None:
        spread = np.minimum(spread, cfg.max_points)

    index_utc = pd.to_datetime(frame.index, utc=True)
    if cfg.hourly_multipliers:
        hours = np.array([cfg.hourly_multipliers.get(int(ts.hour), 1.0) for ts in index_utc], dtype=float)
        spread = spread * hours
    if cfg.weekday_multipliers:
        weekdays = np.array([cfg.weekday_multipliers.get(int(ts.dayofweek), 1.0) for ts in index_utc], dtype=float)
        spread = spread * weekdays

    if cfg.volatility_multiplier:
        volatility = pd.to_numeric(
            frame.get("atr", frame.get("volatility", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        spread = spread * np.maximum(0.0, 1.0 + cfg.volatility_multiplier * volatility.to_numpy(dtype=float))

    if cfg.size_multiplier:
        size_series = pd.to_numeric(
            frame.get("signal_size", frame.get("volume", cfg.reference_size_lot)),
            errors="coerce",
        ).fillna(cfg.reference_size_lot)
        size_factor = np.maximum(size_series.to_numpy(dtype=float), 0.0) / cfg.reference_size_lot
        spread = spread * np.maximum(0.0, 1.0 + cfg.size_multiplier * size_factor)

    return pd.Series(np.maximum(spread, 0.0), index=frame.index, dtype=float)


def apply_configured_spread(
    frame: pd.DataFrame,
    model_parameters: Optional[Dict[str, Any]],
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply explicit spread model settings to a market frame."""
    cfg = _extract_config(model_parameters)
    if cfg is None or frame is None or frame.empty:
        return frame

    provider_spread = pd.to_numeric(frame.get("spread"), errors="coerce")
    if provider_spread is None or not isinstance(provider_spread, pd.Series):
        provider_spread = pd.Series(np.nan, index=frame.index, dtype=float)
    provider_spread = provider_spread.reindex(frame.index).astype(float)

    output = frame.copy()
    static_points = max(cfg.static_points or 0.0, 0.0)
    stochastic_spread = (
        _stochastic_spread(frame, cfg, seed=seed)
        if cfg.stochastic_enabled or cfg.mode in {"stochastic_only", "provider_plus_stochastic"}
        else pd.Series(0.0, index=frame.index, dtype=float)
    )

    if cfg.mode == "provider_only":
        effective = provider_spread.fillna(cfg.provider_fallback_points)
    elif cfg.mode == "stochastic_only":
        effective = stochastic_spread
    elif cfg.mode == "provider_plus_stochastic":
        effective = provider_spread.fillna(cfg.provider_fallback_points) + stochastic_spread
    else:
        # Explicit static mode overrides provider spread.
        effective = pd.Series(static_points, index=frame.index, dtype=float)

    output["spread"] = np.maximum(pd.to_numeric(effective, errors="coerce").fillna(0.0).to_numpy(), 0.0)
    return output


__all__ = ["apply_configured_spread", "SPREAD_MODES", "DEFAULT_DISTRIBUTION"]
