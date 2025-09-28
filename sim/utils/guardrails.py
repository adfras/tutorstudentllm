from __future__ import annotations

"""
Utilities to load Bayesian guardrails and domain-specific talk slopes
emitted by `scripts/bayes/session_bayes_report.py`.

This is an optional helper: callers should handle missing paths gracefully.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json, csv, os


@dataclass
class TokenBands:
    # Raw run-level targets
    opt_run: Optional[float]
    band_low_run: Optional[float]
    band_high_run: Optional[float]
    # Per-step targets
    opt_step: Optional[float]
    band_low_step: Optional[float]
    band_high_step: Optional[float]
    mean_steps: Optional[float]
    # z* info
    z_star_mean: Optional[float]
    z_star_hdi3: Optional[float]
    z_star_hdi97: Optional[float]


def load_guardrails(path: str | None) -> Optional[TokenBands]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tr = obj.get("tokens_run") or {}
        ts = obj.get("tokens_per_step") or {}
        zs = (obj.get("z_star") or {})
        return TokenBands(
            opt_run=_safe_float(tr.get("opt")),
            band_low_run=_safe_float(tr.get("band_low")),
            band_high_run=_safe_float(tr.get("band_high")),
            opt_step=_safe_float(ts.get("opt")),
            band_low_step=_safe_float(ts.get("band_low")),
            band_high_step=_safe_float(ts.get("band_high")),
            mean_steps=_safe_float(ts.get("mean_steps")),
            z_star_mean=_safe_float(zs.get("mean")),
            z_star_hdi3=_safe_float(zs.get("hdi3")),
            z_star_hdi97=_safe_float(zs.get("hdi97")),
        )
    except Exception:
        return None


@dataclass
class TalkSlope:
    domain: str
    mean: float
    sd: float
    hdi_3: float
    hdi_97: float
    prob_positive: float


def load_talk_slopes(path: str | None) -> Dict[str, TalkSlope]:
    out: Dict[str, TalkSlope] = {}
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                dom = str(row.get("domain") or "").strip() or "general"
                try:
                    out[dom] = TalkSlope(
                        domain=dom,
                        mean=float(row.get("mean") or 0.0),
                        sd=float(row.get("sd") or 0.0),
                        hdi_3=float(row.get("hdi_3%") or 0.0),
                        hdi_97=float(row.get("hdi_97%") or 0.0),
                        prob_positive=float(row.get("prob_positive") or 0.5),
                    )
                except Exception:
                    # Skip malformed rows
                    continue
    except Exception:
        return {}
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

