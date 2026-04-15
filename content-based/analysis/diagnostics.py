from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import numpy as np
import pandas as pd

from utils.io import load_train_reviews
from utils.lgbm_known_prefix_deep_features import build_known_prefix_eval_frame, load_known_prefix_embedding_bundle
from utils.split import temporal_train_validation_split


SHORT_SEGMENTS: list[str] = ["2", "3", "4", "5", "2-3", "4-5", "2-5"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _segment_mask(frame: pd.DataFrame, segment: str) -> np.ndarray:
    history_count = frame["history_count"].to_numpy(dtype=np.int32)
    if segment == "2-5":
        return np.isin(history_count, [2, 3, 4, 5])
    if segment == "2":
        return history_count == 2
    if segment == "3":
        return history_count == 3
    if segment == "4":
        return history_count == 4
    if segment == "5":
        return history_count == 5
    if segment == "2-3":
        return np.isin(history_count, [2, 3])
    if segment == "4-5":
        return np.isin(history_count, [4, 5])
    raise ValueError(f"Unsupported segment: {segment}")


def _support_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if 2 <= value <= 5:
        return "2-5"
    if 6 <= value <= 20:
        return "6-20"
    return ">20"


def _quantile_bucket(series: pd.Series, *, prefix: str, q: int = 4) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() == 0:
        return pd.Series(["missing"] * len(series), index=series.index, dtype="string")
    labels = [f"{prefix}_q{idx}" for idx in range(1, q + 1)]
    try:
        bucketed = pd.qcut(valid, q=q, labels=labels, duplicates="drop")
    except ValueError:
        bucketed = pd.Series(["all"] * len(series), index=series.index, dtype="string")
    out = bucketed.astype("string")
    return out.fillna("missing")


def _load_snapshot(snapshot_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    predictions = pd.read_csv(snapshot_root / "known_user_deep_validation_predictions.csv")
    summary = _load_json(snapshot_root / "validation_summary.json")
    config = _load_json(snapshot_root / "known_user_deep_config.json")
    return predictions, summary, config


def _prepare_validation_base(*, data_dir: Path, business_repr_root: Path, max_history_len: int) -> pd.DataFrame:
    train_reviews = load_train_reviews(data_dir)
    train_split, val_split = temporal_train_validation_split(train_reviews, val_size=0.2, timestamp_col="date")
    item_support = train_split.groupby("business_id").size()

    base = val_split[["review_id", "user_id", "business_id", "stars"]].rename(
        columns={"user_id": "user", "business_id": "item", "stars": "rating"}
    )
    base["rating"] = base["rating"].astype(np.float32)
    base["item_train_support"] = base["item"].astype(str).map(item_support).fillna(0).astype(np.int32)
    base["item_status"] = np.where(base["item_train_support"].to_numpy(dtype=np.int32) > 0, "known_item", "new_item")
    base["item_support_bucket"] = base["item_train_support"].map(_support_bucket).astype("string")

    bundle = load_known_prefix_embedding_bundle(business_repr_root.parent)
    prefix_frame = build_known_prefix_eval_frame(
        val_split,
        train_split,
        bundle,
        max_history_len=max_history_len,
        target_history_bands=("2-5",),
    )
    prefix_columns = [
        "review_id",
        "known_prefix_history_count",
        "known_prefix_history_similarity_max",
        "known_prefix_history_similarity_mean",
        "known_prefix_last_item_similarity",
    ]
    prefix_frame = prefix_frame[prefix_columns].copy()
    base = base.merge(prefix_frame, on="review_id", how="left")
    return base


def _attach_snapshot(base: pd.DataFrame, predictions: pd.DataFrame, *, snapshot_name: str) -> pd.DataFrame:
    pred = predictions.copy()
    pred_columns = [
        "review_id",
        "history_band",
        "history_count",
        "history_rating_std",
        "history_positive_share",
        "history_negative_share",
        "incumbent_prediction_raw",
        "deep_prediction_raw",
        "incumbent_branch",
        "alpha",
        "baseline_hat",
        "correction_hat",
        "residual_hat",
        "deep_prediction",
    ]
    pred = pred[pred_columns].copy()
    merged = base.merge(pred, on="review_id", how="left", suffixes=("", "_pred"))
    merged["snapshot"] = snapshot_name
    merged["deep_available"] = np.isfinite(pd.to_numeric(merged["deep_prediction_raw"], errors="coerce"))
    merged["history_count"] = pd.to_numeric(merged["history_count"], errors="coerce").fillna(0).astype(np.int32)
    merged["incumbent_prediction"] = np.floor(pd.to_numeric(merged["incumbent_prediction_raw"], errors="coerce").to_numpy(dtype=np.float32) + 0.5).clip(1, 5).astype(np.float32)
    rounded_deep = pd.Series(
        np.floor(pd.to_numeric(merged["deep_prediction_raw"], errors="coerce").to_numpy(dtype=np.float32) + 0.5).clip(1, 5),
        index=merged.index,
        dtype=np.float32,
    )
    merged["deep_prediction"] = pd.to_numeric(merged["deep_prediction"], errors="coerce").fillna(rounded_deep).astype(np.float32)
    merged["incumbent_abs_error"] = np.abs(merged["rating"].to_numpy(dtype=np.float32) - merged["incumbent_prediction"].to_numpy(dtype=np.float32))
    merged["deep_abs_error"] = np.abs(merged["rating"].to_numpy(dtype=np.float32) - merged["deep_prediction"].to_numpy(dtype=np.float32))
    merged["error_delta_vs_incumbent"] = merged["deep_abs_error"] - merged["incumbent_abs_error"]
    merged["worse_than_incumbent"] = merged["error_delta_vs_incumbent"] > 0
    correction = pd.to_numeric(merged["correction_hat"], errors="coerce")
    residual = pd.to_numeric(merged["residual_hat"], errors="coerce")
    merged["effective_correction"] = correction.fillna(residual).fillna(0.0).astype(np.float32)
    merged["abs_correction"] = np.abs(merged["effective_correction"].to_numpy(dtype=np.float32))
    merged["incumbent_gap_abs"] = np.abs(merged["rating"].to_numpy(dtype=np.float32) - pd.to_numeric(merged["incumbent_prediction_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    short_mask = merged["history_band"].astype(str).eq("2-5") & merged["history_count"].between(2, 5, inclusive="both")
    merged["short_segment"] = pd.Series(
        np.where(
            merged["history_count"].to_numpy(dtype=np.int32) <= 3,
            "2-3",
            "4-5",
        ),
        index=merged.index,
        dtype="string",
    )
    merged.loc[~short_mask, "short_segment"] = pd.NA
    return merged


def _segment_metrics(snapshot_frame: pd.DataFrame) -> pd.DataFrame:
    short_frame = snapshot_frame[
        snapshot_frame["history_band"].astype(str).eq("2-5")
        & snapshot_frame["history_count"].between(2, 5, inclusive="both")
    ].copy()
    rows: list[dict[str, Any]] = []
    for segment in SHORT_SEGMENTS:
        mask = _segment_mask(short_frame, segment)
        subset = short_frame.loc[mask].copy()
        served = subset[subset["deep_available"].to_numpy(dtype=bool)].copy()
        if subset.empty:
            continue
        alpha_values = pd.to_numeric(served["alpha"], errors="coerce").to_numpy(dtype=np.float32) if not served.empty else np.array([], dtype=np.float32)
        rows.append(
            {
                "snapshot": str(snapshot_frame["snapshot"].iloc[0]),
                "segment": segment,
                "total_rows": int(len(subset)),
                "served_rows": int(len(served)),
                "served_pct": float(len(served) / max(len(subset), 1)),
                "incumbent_mae": _mae(served["rating"].to_numpy(dtype=np.float32), served["incumbent_prediction"].to_numpy(dtype=np.float32)),
                "deep_mae": _mae(served["rating"].to_numpy(dtype=np.float32), served["deep_prediction"].to_numpy(dtype=np.float32)),
                "delta_mae": float(served["deep_abs_error"].mean() - served["incumbent_abs_error"].mean()) if not served.empty else float("nan"),
                "alpha_mean": float(alpha_values.mean()) if len(alpha_values) else float("nan"),
                "alpha_p10": float(np.percentile(alpha_values, 10)) if len(alpha_values) else float("nan"),
                "alpha_p50": float(np.percentile(alpha_values, 50)) if len(alpha_values) else float("nan"),
                "alpha_p90": float(np.percentile(alpha_values, 90)) if len(alpha_values) else float("nan"),
                "mean_abs_correction": float(served["abs_correction"].mean()) if not served.empty else float("nan"),
                "worse_pct": float(served["worse_than_incumbent"].mean()) if not served.empty else float("nan"),
                "improved_pct": float((served["error_delta_vs_incumbent"] < 0).mean()) if not served.empty else float("nan"),
                "worse_error_mass_pct": _worse_error_mass_pct(served),
            }
        )
    return pd.DataFrame(rows)


def _worse_error_mass_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    delta = frame["error_delta_vs_incumbent"].to_numpy(dtype=np.float32)
    total_mass = np.abs(delta).sum(dtype=np.float64)
    if total_mass <= 1e-12:
        return 0.0
    return float(np.clip(np.maximum(delta, 0.0).sum(dtype=np.float64) / total_mass, 0.0, 1.0))


def _build_crosscut_frame(snapshot_frame: pd.DataFrame) -> pd.DataFrame:
    short_23 = snapshot_frame[
        snapshot_frame["history_band"].astype(str).eq("2-5")
        & snapshot_frame["history_count"].between(2, 3, inclusive="both")
        & snapshot_frame["deep_available"].to_numpy(dtype=bool)
    ].copy()
    if short_23.empty:
        return pd.DataFrame()

    short_23["incumbent_gap_bucket"] = _quantile_bucket(short_23["incumbent_gap_abs"], prefix="gap")
    short_23["history_variance_bucket"] = _quantile_bucket(short_23["history_rating_std"], prefix="var")
    short_23["prefix_similarity_bucket"] = _quantile_bucket(short_23["known_prefix_history_similarity_mean"], prefix="sim")

    specs = {
        "item_status": short_23["item_status"].astype("string"),
        "item_support_bucket": short_23["item_support_bucket"].astype("string"),
        "incumbent_gap_bucket": short_23["incumbent_gap_bucket"].astype("string"),
        "history_variance_bucket": short_23["history_variance_bucket"].astype("string"),
        "prefix_similarity_bucket": short_23["prefix_similarity_bucket"].astype("string"),
        "exact_prefix_length": short_23["history_count"].astype(str),
    }
    rows: list[dict[str, Any]] = []
    for crosscut_name, series in specs.items():
        for bucket in sorted(series.dropna().astype(str).unique().tolist()):
            mask = series.astype(str) == bucket
            subset = short_23.loc[mask].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    "snapshot": str(snapshot_frame["snapshot"].iloc[0]),
                    "crosscut": crosscut_name,
                    "bucket": bucket,
                    "n_samples": int(len(subset)),
                    "incumbent_mae": float(subset["incumbent_abs_error"].mean()),
                    "deep_mae": float(subset["deep_abs_error"].mean()),
                    "delta_mae": float(subset["deep_abs_error"].mean() - subset["incumbent_abs_error"].mean()),
                    "alpha_mean": float(pd.to_numeric(subset["alpha"], errors="coerce").mean()),
                    "mean_abs_correction": float(subset["abs_correction"].mean()),
                    "worse_pct": float(subset["worse_than_incumbent"].mean()),
                    "improved_pct": float((subset["error_delta_vs_incumbent"] < 0).mean()),
                    "worse_error_mass_pct": _worse_error_mass_pct(subset),
                }
            )
    return pd.DataFrame(rows)


def _hypothesis_summary(snapshot_frame: pd.DataFrame, crosscut_frame: pd.DataFrame) -> dict[str, Any]:
    short_23 = snapshot_frame[
        snapshot_frame["history_band"].astype(str).eq("2-5")
        & snapshot_frame["history_count"].between(2, 3, inclusive="both")
        & snapshot_frame["deep_available"].to_numpy(dtype=bool)
    ].copy()
    if short_23.empty:
        return {"snapshot": str(snapshot_frame["snapshot"].iloc[0]), "hypotheses": {}}

    worse = short_23[short_23["worse_than_incumbent"].to_numpy(dtype=bool)]
    improved = short_23[short_23["error_delta_vs_incumbent"].to_numpy(dtype=np.float32) < 0]
    correction_q = _quantile_bucket(short_23["abs_correction"], prefix="corr")
    correction_rows = []
    for bucket in sorted(correction_q.dropna().astype(str).unique().tolist()):
        subset = short_23.loc[correction_q.astype(str) == bucket]
        correction_rows.append(
            {
                "bucket": bucket,
                "delta_mae": float(subset["deep_abs_error"].mean() - subset["incumbent_abs_error"].mean()),
                "n_samples": int(len(subset)),
            }
        )

    sim_rows = crosscut_frame[crosscut_frame["crosscut"] == "prefix_similarity_bucket"].copy()
    variance_rows = crosscut_frame[crosscut_frame["crosscut"] == "history_variance_bucket"].copy()
    subgroup_spread = float(crosscut_frame["delta_mae"].max() - crosscut_frame["delta_mae"].min()) if not crosscut_frame.empty else float("nan")

    h1_supported = (
        not worse.empty
        and not improved.empty
        and float(worse["abs_correction"].mean()) > float(improved["abs_correction"].mean())
        and _worse_error_mass_pct(short_23) >= 0.55
    )
    h2_supported = (
        len(sim_rows) >= 2
        and len(variance_rows) >= 2
        and float(sim_rows["delta_mae"].max() - sim_rows["delta_mae"].min()) >= 0.01
        and float(variance_rows["delta_mae"].max() - variance_rows["delta_mae"].min()) >= 0.005
    )
    h3_supported = bool(not crosscut_frame.empty and subgroup_spread >= 0.015)

    return {
        "snapshot": str(snapshot_frame["snapshot"].iloc[0]),
        "two_three_rows": int(len(short_23)),
        "two_three_delta_mae": float(short_23["deep_abs_error"].mean() - short_23["incumbent_abs_error"].mean()),
        "hypotheses": {
            "H1_overcorrection": {
                "supported": bool(h1_supported),
                "worse_error_mass_pct": _worse_error_mass_pct(short_23),
                "worse_abs_correction_mean": float(worse["abs_correction"].mean()) if not worse.empty else float("nan"),
                "improved_abs_correction_mean": float(improved["abs_correction"].mean()) if not improved.empty else float("nan"),
                "worse_alpha_mean": float(pd.to_numeric(worse["alpha"], errors="coerce").mean()) if not worse.empty else float("nan"),
                "improved_alpha_mean": float(pd.to_numeric(improved["alpha"], errors="coerce").mean()) if not improved.empty else float("nan"),
                "correction_quantiles": correction_rows,
            },
            "H2_representation_noise": {
                "supported": bool(h2_supported),
                "prefix_similarity_delta_range": float(sim_rows["delta_mae"].max() - sim_rows["delta_mae"].min()) if not sim_rows.empty else float("nan"),
                "history_variance_delta_range": float(variance_rows["delta_mae"].max() - variance_rows["delta_mae"].min()) if not variance_rows.empty else float("nan"),
            },
            "H3_subpopulation_mix": {
                "supported": bool(h3_supported),
                "subgroup_delta_spread": subgroup_spread,
                "best_subgroup_delta": float(crosscut_frame["delta_mae"].min()) if not crosscut_frame.empty else float("nan"),
                "worst_subgroup_delta": float(crosscut_frame["delta_mae"].max()) if not crosscut_frame.empty else float("nan"),
            },
        },
    }


def _wide_segment_comparison(segment_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for segment in SHORT_SEGMENTS:
        subset = segment_frame[segment_frame["segment"] == segment].copy()
        if subset.empty:
            continue
        row: dict[str, Any] = {"segment": segment}
        for _, item in subset.iterrows():
            prefix = str(item["snapshot"])
            row[f"{prefix}_served_rows"] = int(item["served_rows"])
            row[f"{prefix}_incumbent_mae"] = float(item["incumbent_mae"])
            row[f"{prefix}_deep_mae"] = float(item["deep_mae"])
            row[f"{prefix}_delta_mae"] = float(item["delta_mae"])
            row[f"{prefix}_alpha_mean"] = float(item["alpha_mean"])
            row[f"{prefix}_mean_abs_correction"] = float(item["mean_abs_correction"])
            row[f"{prefix}_worse_pct"] = float(item["worse_pct"])
        if {"v3_delta_mae", "v4_delta_mae"}.issubset(row):
            row["v4_minus_v3_delta_mae"] = float(row["v4_delta_mae"] - row["v3_delta_mae"])
        records.append(row)
    return pd.DataFrame(records)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, *, columns: list[str]) -> str:
    if frame.empty:
        return "_Sin datos_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _recommendation_text(v3_hyp: dict[str, Any], v4_hyp: dict[str, Any], summary_v3: dict[str, Any], summary_v4: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    v3_mae = float(summary_v3.get("final_overall_mae", np.nan))
    v4_mae = float(summary_v4.get("final_overall_mae", np.nan))
    if np.isfinite(v3_mae) and np.isfinite(v4_mae) and v3_mae <= v4_mae:
        recs.append("Mantener `v3` como baseline estable: sigue ganando al `v4` en MAE global.")
    if v3_hyp["hypotheses"].get("H1_overcorrection", {}).get("supported") or v4_hyp["hypotheses"].get("H1_overcorrection", {}).get("supported"):
        recs.append("La siguiente iteracion deberia endurecer el gating o limitar la amplitud de correccion en `2-3` antes de abrir mas capacidad.")
    if v4_hyp["hypotheses"].get("H3_subpopulation_mix", {}).get("supported"):
        recs.append("Hay heterogeneidad real dentro de `2-3`; si se mantiene el split, el experto `2-3` debe ser mas conservador y mas regularizado que `4-5`.")
    if v3_hyp["hypotheses"].get("H2_representation_noise", {}).get("supported") or v4_hyp["hypotheses"].get("H2_representation_noise", {}).get("supported"):
        recs.append("El ruido de representacion parece concentrarse en prefijos poco similares o historiales inestables; conviene condicionar la correccion a esas senales.")
    if not recs:
        recs.append("No aparece una frontera suficientemente limpia; la opcion por defecto sigue siendo volver a un experto unico `2-5` mas fuerte al estilo `v3`.")
    return recs


def _build_report(
    *,
    segment_frame: pd.DataFrame,
    wide_frame: pd.DataFrame,
    crosscut_frame: pd.DataFrame,
    v3_hyp: dict[str, Any],
    v4_hyp: dict[str, Any],
    summary_v3: dict[str, Any],
    summary_v4: dict[str, Any],
) -> str:
    segment_table = wide_frame.copy()
    segment_table = segment_table[
        [
            column
            for column in [
                "segment",
                "v3_delta_mae",
                "v4_delta_mae",
                "v4_minus_v3_delta_mae",
                "v3_worse_pct",
                "v4_worse_pct",
                "v3_alpha_mean",
                "v4_alpha_mean",
            ]
            if column in segment_table.columns
        ]
    ]
    top_crosscuts = crosscut_frame.sort_values(["snapshot", "delta_mae", "n_samples"], ascending=[True, True, False]).groupby(["snapshot", "crosscut"], as_index=False).head(2)
    worst_crosscuts = crosscut_frame.sort_values(["snapshot", "delta_mae", "n_samples"], ascending=[True, False, False]).groupby(["snapshot", "crosscut"], as_index=False).head(2)

    recommendations = _recommendation_text(v3_hyp, v4_hyp, summary_v3, summary_v4)
    lines = [
        "# Diagnostico De La Banda 2-5",
        "",
        "## Resumen Ejecutivo",
        "",
        f"- `v3 final_overall_mae = {_format_value(summary_v3.get('final_overall_mae'))}`",
        f"- `v4 final_overall_mae = {_format_value(summary_v4.get('final_overall_mae'))}`",
        f"- `v3 delta 2-5 = {_format_value(next((row['delta_mae'] for row in segment_frame.to_dict(orient='records') if row['snapshot'] == 'v3' and row['segment'] == '2-5'), np.nan))}`",
        f"- `v4 delta 2-5 = {_format_value(next((row['delta_mae'] for row in segment_frame.to_dict(orient='records') if row['snapshot'] == 'v4' and row['segment'] == '2-5'), np.nan))}`",
        "",
        "## Comparativa Por Segmento",
        "",
        _markdown_table(
            segment_table,
            columns=[column for column in segment_table.columns.tolist()],
        ),
        "",
        "## Mejores Y Peores Subgrupos Dentro De 2-3",
        "",
        "### Mejor Respuesta",
        "",
        _markdown_table(
            top_crosscuts,
            columns=["snapshot", "crosscut", "bucket", "n_samples", "delta_mae", "worse_pct", "alpha_mean"],
        ),
        "",
        "### Peor Respuesta",
        "",
        _markdown_table(
            worst_crosscuts,
            columns=["snapshot", "crosscut", "bucket", "n_samples", "delta_mae", "worse_pct", "mean_abs_correction"],
        ),
        "",
        "## Hipotesis",
        "",
        f"- `H1 overcorrection` en `v3`: {'supported' if v3_hyp['hypotheses']['H1_overcorrection']['supported'] else 'not_supported'}",
        f"- `H1 overcorrection` en `v4`: {'supported' if v4_hyp['hypotheses']['H1_overcorrection']['supported'] else 'not_supported'}",
        f"- `H2 representation_noise` en `v3`: {'supported' if v3_hyp['hypotheses']['H2_representation_noise']['supported'] else 'not_supported'}",
        f"- `H2 representation_noise` en `v4`: {'supported' if v4_hyp['hypotheses']['H2_representation_noise']['supported'] else 'not_supported'}",
        f"- `H3 subpopulation_mix` en `v3`: {'supported' if v3_hyp['hypotheses']['H3_subpopulation_mix']['supported'] else 'not_supported'}",
        f"- `H3 subpopulation_mix` en `v4`: {'supported' if v4_hyp['hypotheses']['H3_subpopulation_mix']['supported'] else 'not_supported'}",
        "",
        "## Recomendacion Operativa",
        "",
    ]
    lines.extend([f"- {text}" for text in recommendations])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a diagnostic report for the short known-user band 2-5.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument(
        "--v3-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "known_user_deep_router_v2_eval_v3",
    )
    parser.add_argument(
        "--v4-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "known_user_deep_router_v4_eval_v1",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "known_user_short_band_diagnostic_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    v3_predictions, v3_summary, v3_config = _load_snapshot(args.v3_root)
    v4_predictions, v4_summary, v4_config = _load_snapshot(args.v4_root)
    business_repr_root = Path(v3_config["data_config"]["business_repr_root"])
    max_history_len = int(v3_config["data_config"]["max_history_len"])

    validation_base = _prepare_validation_base(
        data_dir=args.data_dir,
        business_repr_root=business_repr_root,
        max_history_len=max_history_len,
    )
    v3_frame = _attach_snapshot(validation_base, v3_predictions, snapshot_name="v3")
    v4_frame = _attach_snapshot(validation_base, v4_predictions, snapshot_name="v4")

    segment_frame = pd.concat([_segment_metrics(v3_frame), _segment_metrics(v4_frame)], ignore_index=True)
    crosscut_frame = pd.concat([_build_crosscut_frame(v3_frame), _build_crosscut_frame(v4_frame)], ignore_index=True)
    wide_frame = _wide_segment_comparison(segment_frame)

    v3_hyp = _hypothesis_summary(v3_frame, crosscut_frame[crosscut_frame["snapshot"] == "v3"].copy())
    v4_hyp = _hypothesis_summary(v4_frame, crosscut_frame[crosscut_frame["snapshot"] == "v4"].copy())
    report = _build_report(
        segment_frame=segment_frame,
        wide_frame=wide_frame,
        crosscut_frame=crosscut_frame,
        v3_hyp=v3_hyp,
        v4_hyp=v4_hyp,
        summary_v3=v3_summary,
        summary_v4=v4_summary,
    )

    segment_frame.to_csv(save_root / "short_band_segment_metrics.csv", index=False)
    wide_frame.to_csv(save_root / "short_band_segment_comparison.csv", index=False)
    crosscut_frame.to_csv(save_root / "short_band_2_3_crosscuts.csv", index=False)
    v3_frame[v3_frame["history_band"].astype(str) == "2-5"].to_csv(save_root / "v3_short_band_enriched.csv", index=False)
    v4_frame[v4_frame["history_band"].astype(str) == "2-5"].to_csv(save_root / "v4_short_band_enriched.csv", index=False)
    _save_json(
        save_root / "hypothesis_summary.json",
        {
            "v3": v3_hyp,
            "v4": v4_hyp,
            "v3_final_overall_mae": v3_summary.get("final_overall_mae"),
            "v4_final_overall_mae": v4_summary.get("final_overall_mae"),
        },
    )
    _save_json(
        save_root / "analysis_metadata.json",
        {
            "v3_root": str(args.v3_root),
            "v4_root": str(args.v4_root),
            "data_dir": str(args.data_dir),
            "business_repr_root": str(business_repr_root),
            "max_history_len": max_history_len,
            "n_validation_rows": int(len(validation_base)),
        },
    )
    (save_root / "report.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "save_root": str(save_root),
                "segment_metrics_path": str(save_root / "short_band_segment_metrics.csv"),
                "crosscuts_path": str(save_root / "short_band_2_3_crosscuts.csv"),
                "report_path": str(save_root / "report.md"),
                "v3_final_overall_mae": v3_summary.get("final_overall_mae"),
                "v4_final_overall_mae": v4_summary.get("final_overall_mae"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
