

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

AUTHORIZED_USERS = {"jure", "mateo"}

DEFAULT_FEATURE_COLUMNS = [
    "ht_mean",
    "ht_std",
    "dd_mean",
    "dd_std",
    "ud_mean",
    "ud_std",
    "typing_speed",
    "backspace_count",
    "backspace_ratio",
    "total_duration",
    "key_count",
]

DEFAULT_THRESHOLD = 0.60


def normalize_user_id(user_id: str) -> str:
    """Vraca user_id u standardnom obliku."""
    return str(user_id).lower().strip()


@lru_cache(maxsize=1)
def load_feature_columns() -> List[str]:
    """
    Ucitava redoslijed feature stupaca iz models/feature_columns.json.
    Ako datoteka ne postoji, koristi DEFAULT_FEATURE_COLUMNS.
    """
    path = MODELS_DIR / "feature_columns.json"

    if not path.exists():
        return DEFAULT_FEATURE_COLUMNS

    with path.open("r", encoding="utf-8") as file:
        columns = json.load(file)

    if not isinstance(columns, list) or not columns:
        raise ValueError("models/feature_columns.json nije validna lista feature stupaca.")

    return [str(col) for col in columns]


@lru_cache(maxsize=8)
def load_model_bundle(user_id: str) -> Dict[str, Any]:

    user_id = normalize_user_id(user_id)

    if user_id not in AUTHORIZED_USERS:
        raise ValueError(f"Nepoznat ili neautoriziran korisnik: {user_id}")

    model_path = MODELS_DIR / f"{user_id}_model.pkl"
    scaler_path = MODELS_DIR / f"{user_id}_scaler.pkl"
    threshold_path = MODELS_DIR / f"{user_id}_threshold.json"

    missing_files = [
        str(path)
        for path in [model_path, scaler_path]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Nedostaju datoteke modela/scalera: " + ", ".join(missing_files)
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    threshold = DEFAULT_THRESHOLD
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as file:
            threshold_data = json.load(file)

        threshold = float(threshold_data.get("threshold", DEFAULT_THRESHOLD))

    return {
        "user_id": user_id,
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
    }


def prepare_feature_row(feature_vector: Dict[str, Any]) -> pd.DataFrame:

    feature_columns = load_feature_columns()

    missing = [col for col in feature_columns if col not in feature_vector]
    if missing:
        raise ValueError(f"Nedostaju featurei: {missing}")

    row = pd.DataFrame([feature_vector])
    row = row[feature_columns].copy()

    for col in feature_columns:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    if row.isna().any().any():
        bad_columns = row.columns[row.isna().any()].tolist()
        raise ValueError(f"Neki featurei nisu numericki ili su prazni: {bad_columns}")

    return row


def get_positive_score(model: Any, transformed_row: Any) -> float:

    if not hasattr(model, "predict_proba"):
        prediction = int(model.predict(transformed_row)[0])
        return float(prediction)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model ne sadrzi pozitivnu klasu 1.")

    positive_index = classes.index(1)
    score = model.predict_proba(transformed_row)[0][positive_index]

    return float(score)


def predict_for_user(logged_in_user: str, feature_vector: Dict[str, Any]) -> Dict[str, Any]:

    user_id = normalize_user_id(logged_in_user)

    if user_id not in AUTHORIZED_USERS:
        return {
            "accepted": False,
            "score": 0.0,
            "threshold": None,
            "status": "unknown_user",
            "user": user_id,
            "message": "Korisnik nije autoriziran. Dostupni korisnici su: jure, mateo.",
        }

    try:
        bundle = load_model_bundle(user_id)
        row = prepare_feature_row(feature_vector)

        transformed_row = bundle["scaler"].transform(row)
        score = get_positive_score(bundle["model"], transformed_row)

        threshold = float(bundle["threshold"])
        accepted = score >= threshold

        return {
            "accepted": bool(accepted),
            "score": round(score, 4),
            "threshold": threshold,
            "status": "authenticated" if accepted else "suspicious",
            "user": user_id,
        }

    except Exception as exc:
        return {
            "accepted": False,
            "score": 0.0,
            "threshold": None,
            "status": "error",
            "user": user_id,
            "message": str(exc),
        }


def session_status(
    last_results: List[bool],
    max_windows: int = 5,
    warn_if_bad: int = 2,
    lock_if_bad: int = 3,
) -> str:

    recent = list(last_results)[-max_windows:]
    bad_count = recent.count(False)

    if bad_count >= lock_if_bad:
        return "locked"

    if bad_count >= warn_if_bad:
        return "warning"

    return "authenticated"


def clear_model_cache() -> None:

    load_model_bundle.cache_clear()
    load_feature_columns.cache_clear()


if __name__ == "__main__":
    example = {
        "ht_mean": 95.2,
        "ht_std": 14.1,
        "dd_mean": 220.5,
        "dd_std": 80.3,
        "ud_mean": 120.2,
        "ud_std": 60.5,
        "typing_speed": 4.8,
        "backspace_count": 1,
        "backspace_ratio": 0.025,
        "total_duration": 8.3,
        "key_count": 40,
    }

    print(predict_for_user("jure", example))
    print(predict_for_user("mateo", example))
