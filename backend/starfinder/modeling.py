from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.starfinder.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ]
    )


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    confusion_matrix: Any
    classification_report: Dict[str, Any]


def train_and_select_best(df: pd.DataFrame, random_state: int = 42) -> Tuple[ModelResult, Dict[str, Any]]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = make_preprocessor()

    candidates: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=800, n_jobs=None),
        "random_forest": RandomForestClassifier(n_estimators=500, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=9),
    }

    results: Dict[str, Any] = {}
    best: ModelResult | None = None

    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        candidate = ModelResult(
            name=name,
            pipeline=pipeline,
            accuracy=acc,
            confusion_matrix=cm,
            classification_report=report,
        )

        if best is None:
            best = candidate
        else:
            if candidate.accuracy > best.accuracy:
                best = candidate
            elif candidate.accuracy == best.accuracy:
                cand_f1 = float(candidate.classification_report["macro avg"]["f1-score"])
                best_f1 = float(best.classification_report["macro avg"]["f1-score"])
                if cand_f1 > best_f1:
                    best = candidate

    assert best is not None
    return best, results

