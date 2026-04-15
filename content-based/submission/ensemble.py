from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _discover_submission_paths(
    *,
    artifacts_root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[Path]:
    paths = sorted(artifacts_root.rglob("submission.csv"))
    selected: list[Path] = []
    for path in paths:
        text = str(path).lower()
        if include_patterns and not all(pattern.lower() in text for pattern in include_patterns):
            continue
        if any(pattern.lower() in text for pattern in exclude_patterns):
            continue
        selected.append(path)
    return selected


def _load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["review_id", "stars"], low_memory=False)
    frame["review_id"] = frame["review_id"].astype(str)
    frame["stars"] = pd.to_numeric(frame["stars"], errors="raise").astype(np.float32)
    return frame


def _mode_rounded(values: np.ndarray) -> np.ndarray:
    rounded = _round_half_up(values.astype(np.float32)).clip(1, 5)
    out = np.zeros(len(rounded), dtype=np.int32)
    for idx, row in enumerate(rounded):
        counts = np.bincount(row.astype(np.int32), minlength=6)
        best = np.flatnonzero(counts == counts.max())
        if len(best) == 1:
            out[idx] = int(best[0])
            continue
        median_tiebreak = int(_round_half_up(np.array([np.median(values[idx])], dtype=np.float32))[0])
        out[idx] = median_tiebreak if median_tiebreak in best else int(best[-1])
    return out


def _trimmed_mean(values: np.ndarray, trim_ratio: float) -> np.ndarray:
    if values.shape[1] <= 2 or trim_ratio <= 0:
        return values.mean(axis=1)
    trim_count = int(np.floor(values.shape[1] * trim_ratio))
    if trim_count == 0 or (2 * trim_count) >= values.shape[1]:
        return values.mean(axis=1)
    sorted_values = np.sort(values, axis=1)
    return sorted_values[:, trim_count:-trim_count].mean(axis=1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "artifacts"
    parser = argparse.ArgumentParser(description="Ensemble all submission.csv files found under artifacts using robust aggregations.")
    parser.add_argument("--artifacts-root", type=Path, default=root)
    parser.add_argument("--save-root", type=Path, default=root / "artifact_submission_ensemble_v1")
    parser.add_argument("--include", action="append", default=[], help="Substring filter; may be passed multiple times.")
    parser.add_argument("--exclude", action="append", default=[], help="Substring exclusion filter; may be passed multiple times.")
    parser.add_argument("--method", choices=["median", "mean", "trimmed_mean", "mode"], default="median")
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = args.save_path or (save_root / "submission.csv")

    submission_paths = _discover_submission_paths(
        artifacts_root=args.artifacts_root,
        include_patterns=list(args.include),
        exclude_patterns=list(args.exclude),
    )
    if len(submission_paths) < 2:
        raise RuntimeError("Need at least two submissions to build an ensemble.")

    base = _load_submission(submission_paths[0]).rename(columns={"stars": "stars_000"})
    used_paths = [submission_paths[0]]
    for index, path in enumerate(submission_paths[1:], start=1):
        current = _load_submission(path).rename(columns={"stars": f"stars_{index:03d}"})
        merged = base.merge(current, on="review_id", how="inner", validate="one_to_one")
        if len(merged) != len(base):
            raise RuntimeError(f"Submission {path} does not align perfectly by review_id.")
        base = merged
        used_paths.append(path)

    score_columns = [column for column in base.columns if column.startswith("stars_")]
    score_matrix = base[score_columns].to_numpy(dtype=np.float32)

    if args.method == "median":
        ensemble_raw = np.median(score_matrix, axis=1)
    elif args.method == "mean":
        ensemble_raw = score_matrix.mean(axis=1)
    elif args.method == "trimmed_mean":
        ensemble_raw = _trimmed_mean(score_matrix, trim_ratio=float(args.trim_ratio))
    else:
        ensemble_raw = _mode_rounded(score_matrix).astype(np.float32)

    ensemble_rounded = _round_half_up(ensemble_raw).clip(1, 5).astype(np.int32)
    submission = pd.DataFrame({"review_id": base["review_id"].astype(str), "stars": ensemble_rounded})
    submission.to_csv(save_path, index=False)

    disagreement = pd.DataFrame(
        {
            "review_id": base["review_id"].astype(str),
            "prediction_min": score_matrix.min(axis=1).astype(np.float32),
            "prediction_max": score_matrix.max(axis=1).astype(np.float32),
            "prediction_range": (score_matrix.max(axis=1) - score_matrix.min(axis=1)).astype(np.float32),
            "prediction_std": score_matrix.std(axis=1).astype(np.float32),
            "ensemble_raw": ensemble_raw.astype(np.float32),
            "ensemble_rounded": ensemble_rounded.astype(np.int32),
        }
    )
    disagreement.to_csv(save_root / "submission_disagreement.csv", index=False)

    member_summary = pd.DataFrame(
        {
            "member_index": list(range(len(used_paths))),
            "path": [str(path) for path in used_paths],
            "mean_prediction": [float(score_matrix[:, idx].mean()) for idx in range(score_matrix.shape[1])],
            "std_prediction": [float(score_matrix[:, idx].std()) for idx in range(score_matrix.shape[1])],
        }
    )
    member_summary.to_csv(save_root / "ensemble_members.csv", index=False)

    summary = {
        "artifacts_root": str(args.artifacts_root),
        "save_path": str(save_path),
        "method": args.method,
        "trim_ratio": float(args.trim_ratio),
        "n_members": len(used_paths),
        "members": [str(path) for path in used_paths],
        "n_rows": int(len(submission)),
        "prediction_min": int(submission["stars"].min()),
        "prediction_max": int(submission["stars"].max()),
        "prediction_mean": float(submission["stars"].mean()),
        "mean_member_std": float(disagreement["prediction_std"].mean()),
        "mean_member_range": float(disagreement["prediction_range"].mean()),
    }
    (save_root / "ensemble_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
