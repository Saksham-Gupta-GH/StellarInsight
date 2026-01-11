from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.starfinder.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    REQUIRED_FEATURES,
)


@dataclass(frozen=True)
class DomainSpec:
    numeric_ranges: Dict[str, Tuple[float, float]]
    allowed_colors: List[str]
    allowed_spectral_classes: List[str]

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "DomainSpec":
        numeric_ranges = {
            key: (float(value[0]), float(value[1])) for key, value in payload["numeric_ranges"].items()
        }
        return DomainSpec(
            numeric_ranges=numeric_ranges,
            allowed_colors=list(payload["allowed_colors"]),
            allowed_spectral_classes=list(payload["allowed_spectral_classes"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_ranges": {k: [float(v[0]), float(v[1])] for k, v in self.numeric_ranges.items()},
            "allowed_colors": list(self.allowed_colors),
            "allowed_spectral_classes": list(self.allowed_spectral_classes),
        }


def build_domain_spec(df: pd.DataFrame) -> DomainSpec:
    numeric_ranges: Dict[str, Tuple[float, float]] = {}
    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce")
        numeric_ranges[col] = (float(series.min()), float(series.max()))

    allowed_colors = sorted(set(df["Color"].astype(str).dropna().unique().tolist()))
    allowed_spectral_classes = sorted(set(df["Spectral_Class"].astype(str).dropna().unique().tolist()))

    return DomainSpec(
        numeric_ranges=numeric_ranges,
        allowed_colors=allowed_colors,
        allowed_spectral_classes=allowed_spectral_classes,
    )


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    out_of_domain: bool
    message: Optional[str]
    features: Optional[pd.DataFrame]


def _missing_required(payload: Dict[str, Any]) -> List[str]:
    missing = []
    for key in REQUIRED_FEATURES:
        if key not in payload:
            missing.append(key)
    return missing


def validate_payload(payload: Dict[str, Any], domain_spec: DomainSpec) -> ValidationResult:
    missing = _missing_required(payload)
    if missing:
        return ValidationResult(
            ok=False,
            out_of_domain=False,
            message=f"Missing required fields: {', '.join(missing)}",
            features=None,
        )

    parsed: Dict[str, Any] = {}
    for col in NUMERIC_FEATURES:
        try:
            value = float(payload[col])
        except Exception:
            return ValidationResult(
                ok=False,
                out_of_domain=False,
                message=f"Invalid numeric value for '{col}'.",
                features=None,
            )

        if not np.isfinite(value):
            return ValidationResult(
                ok=False,
                out_of_domain=False,
                message=f"Invalid numeric value for '{col}'.",
                features=None,
            )

        low, high = domain_spec.numeric_ranges[col]
        if value < low or value > high:
            return ValidationResult(ok=False, out_of_domain=True, message=None, features=None)

        parsed[col] = value

    color = str(payload["Color"]).strip()
    spectral = str(payload["Spectral_Class"]).strip()

    if color not in domain_spec.allowed_colors:
        return ValidationResult(ok=False, out_of_domain=True, message=None, features=None)
    if spectral not in domain_spec.allowed_spectral_classes:
        return ValidationResult(ok=False, out_of_domain=True, message=None, features=None)

    parsed["Color"] = color
    parsed["Spectral_Class"] = spectral

    return ValidationResult(
        ok=True,
        out_of_domain=False,
        message=None,
        features=pd.DataFrame([parsed], columns=REQUIRED_FEATURES),
    )

