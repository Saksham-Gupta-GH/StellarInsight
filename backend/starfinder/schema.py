from __future__ import annotations

from typing import Dict, Tuple

NUMERIC_FEATURES: Tuple[str, ...] = ("Temperature", "L", "R", "A_M")
CATEGORICAL_FEATURES: Tuple[str, ...] = ("Color", "Spectral_Class")
REQUIRED_FEATURES: Tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "Type"

LABEL_ID_TO_NAME: Dict[str, str] = {
    "0": "Red Dwarf",
    "1": "Brown Dwarf",
    "2": "White Dwarf",
    "3": "Main Sequence",
    "4": "Super Giants",
    "5": "Hyper Giants",
}

PREDICTION_WARNING_MESSAGE = (
    "The parameters provided don't match the stellar profiles in our dataset. "
    "This suggests we might be looking at a compact object or an exotic phenomenon "
    "that requires further analysis beyond this model's scope."
)

