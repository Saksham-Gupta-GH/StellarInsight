from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from backend.starfinder.domain import build_domain_spec
from backend.starfinder.modeling import train_and_select_best


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _project_root()
    df = pd.read_csv(root / "Stars.csv")

    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best, all_results = train_and_select_best(df)
    joblib.dump(best.pipeline, artifacts_dir / "model.joblib")

    domain_spec = build_domain_spec(df)
    (artifacts_dir / "domain.json").write_text(
        json.dumps(domain_spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_model": best.name,
                "all_models": all_results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {artifacts_dir / 'model.joblib'}")
    print(f"Saved domain: {artifacts_dir / 'domain.json'}")
    print(f"Saved metrics: {artifacts_dir / 'metrics.json'}")
    print(f"Best model: {best.name} (accuracy={best.accuracy:.4f})")


if __name__ == "__main__":
    main()

