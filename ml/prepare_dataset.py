#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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


IGNORED_KEYS = {
    "Shift", "Control", "Alt", "Meta", "CapsLock",
    "Tab", "Escape", "ArrowLeft", "ArrowRight",
    "ArrowUp", "ArrowDown",
}


def load_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        return [data]

    raise ValueError(f"Nepodrzan JSON format u datoteci: {path}")


def collect_input_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        else:
            print(f"[UPOZORENJE] Preskacem jer nije JSON datoteka/direktorij: {path}")

    unique_files = []
    seen = set()
    for file in files:
        resolved = file.resolve()
        if resolved not in seen:
            unique_files.append(file)
            seen.add(resolved)

    return unique_files


def clean_event(event: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        key = str(event["key"])
        keydown = float(event["keydown"])
        keyup = float(event["keyup"])
    except (KeyError, TypeError, ValueError):
        return None

    if key in IGNORED_KEYS:
        return None

    if not np.isfinite(keydown) or not np.isfinite(keyup):
        return None

    if keyup < keydown:
        return None

    return {
        "key": key,
        "keydown": keydown,
        "keyup": keyup,
    }


def clean_sample(sample: Dict[str, Any], new_sample_id: int) -> Dict[str, Any] | None:
    user_id = str(sample.get("user_id", "")).strip()
    if not user_id:
        user_id = "unknown"

    try:
        label = int(sample.get("label"))
    except (TypeError, ValueError):
        return None

    if label not in (0, 1):
        return None

    raw_events = sample.get("events", [])
    if not isinstance(raw_events, list):
        return None

    events = []
    for event in raw_events:
        if isinstance(event, dict):
            cleaned = clean_event(event)
            if cleaned is not None:
                events.append(cleaned)

    events.sort(key=lambda e: e["keydown"])

    if len(events) == 0:
        return None

    return {
        "sample_id": new_sample_id,
        "original_sample_id": sample.get("sample_id"),
        "user_id": user_id,
        "label": label,
        "events": events,
        "backspace_count": int(sample.get("backspace_count", 0) or 0),
        "created_at": sample.get("created_at"),
    }


def merge_raw_samples(input_files: List[Path]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    next_id = 1

    for path in input_files:
        print(f"[INFO] Ucitavam: {path}")
        samples = load_json_file(path)

        for sample in samples:
            if not isinstance(sample, dict):
                continue

            cleaned = clean_sample(sample, next_id)
            if cleaned is None:
                continue

            merged.append(cleaned)
            next_id += 1

    return merged


def safe_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else 0.0


def safe_std(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=0)) if len(arr) else 0.0


def extract_features_from_events(
    events: List[Dict[str, Any]],
    user_id: str,
    label: int,
    sample_id: int,
    window_id: int = 0,
) -> Dict[str, Any] | None:

    if len(events) < 2:
        return None

    keydowns = np.array([float(e["keydown"]) for e in events], dtype=float)
    keyups = np.array([float(e["keyup"]) for e in events], dtype=float)

    ht = keyups - keydowns
    dd = keydowns[1:] - keydowns[:-1]
    ud = keydowns[1:] - keyups[:-1]

    start_time = float(keydowns[0])
    end_time = float(max(keyups[-1], keydowns[-1]))
    total_duration_ms = max(end_time - start_time, 1.0)
    total_duration_sec = total_duration_ms / 1000.0

    key_count = len(events)
    backspace_count = sum(1 for e in events if str(e.get("key")) == "Backspace")
    backspace_ratio = backspace_count / key_count if key_count else 0.0
    typing_speed = key_count / total_duration_sec if total_duration_sec > 0 else 0.0

    return {
        "sample_id": sample_id,
        "window_id": window_id,
        "user_id": user_id,
        "ht_mean": safe_mean(ht),
        "ht_std": safe_std(ht),
        "dd_mean": safe_mean(dd),
        "dd_std": safe_std(dd),
        "ud_mean": safe_mean(ud),
        "ud_std": safe_std(ud),
        "typing_speed": float(typing_speed),
        "backspace_count": int(backspace_count),
        "backspace_ratio": float(backspace_ratio),
        "total_duration": float(total_duration_sec),
        "key_count": int(key_count),
        "label": int(label),
    }


def sample_to_feature_rows(
    sample: Dict[str, Any],
    window_size: int,
    stride: int,
    min_events: int,
    split_windows: bool,
) -> List[Dict[str, Any]]:
    events = sample["events"]
    rows: List[Dict[str, Any]] = []

    if split_windows:
        window_id = 0
        for start in range(0, len(events) - window_size + 1, stride):
            window_events = events[start:start + window_size]
            if len(window_events) < min_events:
                continue

            row = extract_features_from_events(
                events=window_events,
                user_id=sample["user_id"],
                label=sample["label"],
                sample_id=sample["sample_id"],
                window_id=window_id,
            )
            if row is not None:
                rows.append(row)
                window_id += 1
    else:
        if len(events) >= min_events:
            row = extract_features_from_events(
                events=events,
                user_id=sample["user_id"],
                label=sample["label"],
                sample_id=sample["sample_id"],
                window_id=0,
            )
            if row is not None:
                rows.append(row)

    return rows


def build_features_dataframe(
    samples: List[Dict[str, Any]],
    window_size: int,
    stride: int,
    min_events: int,
    split_windows: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sample in samples:
        rows.extend(
            sample_to_feature_rows(
                sample=sample,
                window_size=window_size,
                stride=stride,
                min_events=min_events,
                split_windows=split_windows,
            )
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    ordered_columns = ["sample_id", "window_id", "user_id"] + FEATURE_COLUMNS + ["label"]
    df = df[ordered_columns]

    numeric_cols = FEATURE_COLUMNS + ["label"]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spoji raw_samples JSON datoteke i generiraj features.csv.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Popis JSON datoteka ili direktorija koji sadrze JSON datoteke.",
    )
    parser.add_argument(
        "--output-raw",
        default="data/raw_samples.json",
        help="Putanja za spojeni raw_samples.json.",
    )
    parser.add_argument(
        "--output-features",
        default="data/features.csv",
        help="Putanja za generirani features.csv.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=40,
        help="Velicina prozora u broju tipki.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=40,
        help="Pomak prozora. 40 znaci bez preklapanja, 20 znaci 50%% preklapanja.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=30,
        help="Minimalan broj eventa potreban za feature redak.",
    )
    parser.add_argument(
        "--no-split-windows",
        action="store_true",
        help="Ako je ukljuceno, svaki raw sample postaje jedan feature redak.",
    )

    args = parser.parse_args()

    input_files = collect_input_files(args.input)
    if not input_files:
        raise SystemExit("[GRESKA] Nema JSON datoteka za ucitavanje.")

    raw_samples = merge_raw_samples(input_files)
    if not raw_samples:
        raise SystemExit("[GRESKA] Nema validnih raw uzoraka nakon ciscenja.")

    output_raw = Path(args.output_raw)
    save_json(raw_samples, output_raw)

    features_df = build_features_dataframe(
        samples=raw_samples,
        window_size=args.window_size,
        stride=args.stride,
        min_events=args.min_events,
        split_windows=not args.no_split_windows,
    )

    if features_df.empty:
        raise SystemExit("[GRESKA] features.csv je prazan. Provjeri broj eventa u uzorcima.")

    output_features = Path(args.output_features)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_features, index=False, encoding="utf-8")

    print("\n[OK] Spajanje i pretvorba zavrseni.")
    print(f"[OK] Spojeni raw samples: {output_raw}")
    print(f"[OK] Features CSV: {output_features}")
    print(f"[INFO] Broj raw uzoraka: {len(raw_samples)}")
    print(f"[INFO] Broj feature redaka: {len(features_df)}")
    print("\n[INFO] Broj feature redaka po labeli:")
    print(features_df["label"].value_counts().sort_index().to_string())
    print("\n[INFO] Broj feature redaka po korisniku:")
    print(features_df["user_id"].value_counts().to_string())


if __name__ == "__main__":
    main()
