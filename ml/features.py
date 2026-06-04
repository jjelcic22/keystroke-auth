

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List

import numpy as np


IGNORED_KEYS = {
    "Shift", "Control", "Alt", "Meta", "CapsLock",
    "Tab", "Escape", "ArrowLeft", "ArrowRight",
    "ArrowUp", "ArrowDown",
}


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


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else 0.0


def _safe_std(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=0)) if len(arr) else 0.0


def _clean_paired_event(event: Dict[str, Any]) -> Dict[str, Any] | None:
    """Validira event formata {"key", "keydown", "keyup"}."""
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


def _convert_raw_type_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    down_times: dict[str, deque[float]] = defaultdict(deque)
    paired_events: List[Dict[str, Any]] = []

    sorted_events = sorted(
        events,
        key=lambda e: float(e.get("timestamp", 0.0) or 0.0)
    )

    for event in sorted_events:
        key = str(event.get("key", ""))
        event_type = str(event.get("type", ""))
        timestamp_raw = event.get("timestamp")

        if key in IGNORED_KEYS:
            continue

        try:
            timestamp = float(timestamp_raw)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(timestamp):
            continue

        if event_type == "keydown":
            down_times[key].append(timestamp)

        elif event_type == "keyup":
            if down_times[key]:
                keydown = down_times[key].popleft()
                keyup = timestamp

                if keyup >= keydown:
                    paired_events.append({
                        "key": key,
                        "keydown": keydown,
                        "keyup": keyup,
                    })

    paired_events.sort(key=lambda e: e["keydown"])
    return paired_events


def normalize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prima listu eventa iz frontenda i vraća listu paired eventa.
    """
    if not isinstance(events, list):
        raise ValueError("events mora biti lista.")

    # Ako barem jedan event ima keydown/keyup, pretpostavi novi paired format.
    has_paired_format = any(
        isinstance(e, dict) and "keydown" in e and "keyup" in e
        for e in events
    )

    if has_paired_format:
        cleaned = []
        for event in events:
            if isinstance(event, dict):
                cleaned_event = _clean_paired_event(event)
                if cleaned_event is not None:
                    cleaned.append(cleaned_event)

        cleaned.sort(key=lambda e: e["keydown"])
        return cleaned

    # Inače pokušaj stari type/timestamp format.
    return _convert_raw_type_events(events)


def extract_features_from_window(
    events: List[Dict[str, Any]],
    backspace_count: int = 0,
    min_events: int = 30,
) -> Dict[str, Any]:

    paired_events = normalize_events(events)

    if len(paired_events) < min_events:
        raise ValueError(
            f"Premalo validnih key eventa: {len(paired_events)}. "
            f"Minimalno je potrebno {min_events}."
        )

    keydowns = np.array([float(e["keydown"]) for e in paired_events], dtype=float)
    keyups = np.array([float(e["keyup"]) for e in paired_events], dtype=float)

    ht = keyups - keydowns
    dd = keydowns[1:] - keydowns[:-1]
    ud = keydowns[1:] - keyups[:-1]

    start_time = float(keydowns[0])
    end_time = float(max(keyups[-1], keydowns[-1]))
    total_duration_ms = max(end_time - start_time, 1.0)
    total_duration_sec = total_duration_ms / 1000.0

    key_count = len(paired_events)

    # Ako frontend šalje backspace_count, koristi ga.
    # Ako ne šalje, izračunaj iz eventa.
    counted_backspaces = sum(1 for e in paired_events if str(e.get("key")) == "Backspace")
    final_backspace_count = int(backspace_count) if backspace_count is not None else counted_backspaces

    # Sigurnosno: ako je frontend poslao 0, a eventovi imaju Backspace, uzmi veći broj.
    final_backspace_count = max(final_backspace_count, counted_backspaces)

    backspace_ratio = final_backspace_count / key_count if key_count else 0.0
    typing_speed = key_count / total_duration_sec if total_duration_sec > 0 else 0.0

    feature_vector = {
        "ht_mean": _safe_mean(ht),
        "ht_std": _safe_std(ht),
        "dd_mean": _safe_mean(dd),
        "dd_std": _safe_std(dd),
        "ud_mean": _safe_mean(ud),
        "ud_std": _safe_std(ud),
        "typing_speed": float(typing_speed),
        "backspace_count": int(final_backspace_count),
        "backspace_ratio": float(backspace_ratio),
        "total_duration": float(total_duration_sec),
        "key_count": int(key_count),
    }

    return feature_vector
