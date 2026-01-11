from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.starfinder.schema import NUMERIC_FEATURES, TARGET_COLUMN


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = df.describe(include="all").transpose()
    summary.to_csv(output_dir / "summary_statistics.csv")

    class_dist = df[TARGET_COLUMN].value_counts().sort_index()
    class_dist.to_csv(output_dir / "class_distribution.csv", header=["count"])

    numeric_df = df[list(NUMERIC_FEATURES)].copy()
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", square=True)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(df["Temperature"], df["L"], c=df[TARGET_COLUMN], cmap="viridis", s=35, alpha=0.9)
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Luminosity (L / Lo) [log scale]")
    plt.title("Hertzsprung–Russell Diagram")
    plt.tight_layout()
    plt.savefig(output_dir / "hr_diagram.png", dpi=200)
    plt.close()

