from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.starfinder.eda import run_eda


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _project_root()
    df = pd.read_csv(root / "Stars.csv")
    run_eda(df=df, output_dir=root / "reports")
    print(f"Saved EDA outputs to: {root / 'reports'}")


if __name__ == "__main__":
    main()

