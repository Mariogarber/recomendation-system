from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import numpy as np
import pandas as pd


SHORT_SEGMENTS = ("2", "3", "4", "5", "2-3", "4-5")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_mask(frame: pd.DataFrame, segment: str) -> np.ndarray:
    counts = frame["history_count"].to_numpy(dtype=np.int32)
    if segment == "2":
        return counts == 2
    if segment == "3":
        return counts == 3
    if segment == "4":
        return counts == 4
    if segment == "5":
        return counts == 5
    if segment == "2-3":
        return np.isin(counts, [2, 3])
    if segment == "4-5":
        return np.isin(counts, [4, 5])
    raise ValueError(f"Unsupported segment: {segment}")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).clip(1.0, 5.0).astype(np.float32)


def _compute_short_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    short = frame[frame["history_band"].astype(str).eq("2-5")].copy()
    rows: list[dict[str, Any]] = []
    for segment in SHORT_SEGMENTS:
        mask = _segment_mask(short, segment)
        subset = short.loc[mask].copy()
        if subset.empty:
            continue
        y_true = subset["rating"].to_numpy(dtype=np.float32)
        y_pred = subset["pred_router_rounded"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "history_count_segment": segment,
                "n_samples": int(len(subset)),
                "mae": _mae(y_true, y_pred),
            }
        )
    return rows


def _load_v3_segment_reference(v3_root: Path) -> dict[str, float]:
    frame = pd.read_csv(v3_root / "known_user_deep_validation_predictions.csv")
    frame = frame[frame["history_band"].astype(str).eq("2-5")].copy()
    rounded = pd.Series(
        np.floor(pd.to_numeric(frame["deep_prediction_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) + 0.5).clip(1.0, 5.0),
        index=frame.index,
        dtype=np.float32,
    )
    frame["deep_prediction_rounded"] = pd.to_numeric(frame["deep_prediction"], errors="coerce").fillna(rounded)
    out: dict[str, float] = {}
    for segment in SHORT_SEGMENTS:
        mask = _segment_mask(frame, segment)
        subset = frame.loc[mask]
        if subset.empty:
            continue
        out[segment] = _mae(subset["rating"].to_numpy(dtype=np.float32), subset["deep_prediction_rounded"].to_numpy(dtype=np.float32))
    return out


def _augment_validation_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    augmented = frame.copy()
    if "history_count" not in augmented.columns:
        augmented["history_count"] = 0
    augmented["history_count"] = pd.to_numeric(augmented["history_count"], errors="coerce").fillna(0).astype(np.int32)
    if "pred_router_rounded" not in augmented.columns and "pred_router_raw" in augmented.columns:
        augmented["pred_router_rounded"] = _round_half_up(augmented["pred_router_raw"].to_numpy(dtype=np.float32))
    return augmented


def _run_transition_experiment(*, save_root: Path, passthrough_args: list[str]) -> None:
    script_path = Path(__file__).resolve().parents[2] / "train_transition.py"
    command = [sys.executable, str(script_path), "--save-root", str(save_root), *passthrough_args]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a feature-first short-history router experiment and enrich its summary.")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_feature_first_short_router_v1",
    )
    parser.add_argument(
        "--baseline-v3-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "known_user_deep_router_v2_eval_v3",
    )
    args, unknown = parser.parse_known_args()
    setattr(args, "passthrough_args", unknown)
    return args


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    _run_transition_experiment(save_root=save_root, passthrough_args=list(getattr(args, "passthrough_args", [])))

    validation_summary_path = save_root / "validation_summary.json"
    validation_predictions_path = save_root / "validation_predictions.csv"
    summary = _load_json(validation_summary_path)
    validation_predictions = _augment_validation_predictions(pd.read_csv(validation_predictions_path))

    short_metrics = _compute_short_metrics(validation_predictions)
    v3_reference = _load_v3_segment_reference(args.baseline_v3_root)
    comparison_rows = []
    for row in short_metrics:
        segment = str(row["history_count_segment"])
        v3_mae = float(v3_reference.get(segment, np.nan))
        comparison_rows.append(
            {
                **row,
                "baseline_v3_mae": v3_mae,
                "mae_delta_vs_v3": float(row["mae"] - v3_mae) if np.isfinite(v3_mae) else float("nan"),
            }
        )

    feature_first_summary = {
        "baseline_reference": str(args.baseline_v3_root),
        "short_history_metrics": comparison_rows,
        "feature_first_assumption": "Enhanced raw and known-prefix features are active through the shared builders.",
    }
    summary["feature_first_short_history_eval"] = feature_first_summary
    validation_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(save_root / "short_history_vs_v3.csv", index=False)

    print(
        json.dumps(
            {
                "save_root": str(save_root),
                "validation_summary": str(validation_summary_path),
                "short_history_vs_v3": str(save_root / "short_history_vs_v3.csv"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
