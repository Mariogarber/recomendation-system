from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Template
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from utils.io import canonicalize_reviews, get_default_data_dir, load_train_reviews, load_users
from utils.split import temporal_train_validation_split


SEED = 42
MACRO_CATEGORY_PRIORITY = [
    "Restaurants",
    "Food",
    "Shopping",
    "Beauty & Spas",
    "Health & Medical",
    "Home Services",
    "Local Services",
    "Nightlife",
    "Automotive",
    "Active Life",
    "Event Planning & Services",
    "Arts & Entertainment",
    "Hotels & Travel",
    "Professional Services",
]
PALETTE = [
    "#0f4c5c",
    "#e36414",
    "#6a994e",
    "#bc4749",
    "#7b2cbf",
    "#ffb703",
    "#3a86ff",
    "#8338ec",
    "#fb5607",
    "#2a9d8f",
    "#ef476f",
    "#577590",
]


@dataclass(slots=True)
class LoadedArtifacts:
    business_ids: pd.Series
    business_full: sparse.csr_matrix
    business_deep: np.ndarray
    business_table: pd.DataFrame
    business_summary: dict[str, Any]
    business_block_summary: pd.DataFrame
    user_manual_ids: pd.Series
    user_manual_profile: sparse.csr_matrix
    user_manual_table: pd.DataFrame
    user_manual_summary: dict[str, Any]
    user_deep_ids: pd.Series
    user_deep: np.ndarray
    user_deep_table: pd.DataFrame
    user_deep_summary: dict[str, Any]


class ArtifactLoadError(RuntimeError):
    pass


TABULAR_SUFFIXES = (".csv", ".parquet")
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "business_table": ("business_id", "kept_category_tokens", "city_bucket"),
    "business_block_summary": ("start_index", "coverage"),
    "user_manual_table": ("user_id",),
    "user_deep_table": (
        "user_id",
        "embedding_source",
        "history_band",
        "history_count_train",
        "metadata__tenure_days",
        "metadata__fans_log1p_z",
        "metadata__useful_log1p_z",
        "metadata__cool_log1p_z",
        "metadata__elite_any",
    ),
    "users_table": ("user_id", "friends"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed HTML report for competition embeddings.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "competition_embeddings_v1",
    )
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "competition_embeddings_v1" / "report",
    )
    parser.add_argument("--top-k-neighbors", type=int, default=20)
    parser.add_argument("--business-anchor-per-category", type=int, default=80)
    parser.add_argument("--sample-business-pca", type=int, default=4000)
    parser.add_argument("--sample-user-consistency", type=int, default=1500)
    parser.add_argument("--sample-user-history-clustering", type=int, default=60000)
    parser.add_argument("--sample-cold-clustering", type=int, default=30000)
    parser.add_argument("--sample-friend-anchors", type=int, default=8000)
    parser.add_argument("--friend-pairs-per-anchor", type=int, default=3)
    return parser.parse_args()


def load_sparse_npz(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr().astype(np.float32)


def load_dense_npz(path: Path) -> np.ndarray:
    return np.load(path)["embeddings"].astype(np.float32)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ArtifactLoadError(f"Unsupported tabular extension for {path}")


def resolve_tabular_path(directory: Path, stem: str) -> Path | None:
    for suffix in TABULAR_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_required_tabular(directory: Path, stem: str, label: str, missing: list[str]) -> pd.DataFrame | None:
    path = resolve_tabular_path(directory, stem)
    if path is None:
        candidates = ", ".join(f"{stem}{suffix}" for suffix in TABULAR_SUFFIXES)
        missing.append(f"{label}: expected one of {candidates}")
        return None
    try:
        return read_tabular_file(path)
    except Exception as exc:  # pragma: no cover - defensive, surfaced to caller
        raise ArtifactLoadError(f"Failed to load {label} from {path}: {exc}") from exc


def load_required_json(path: Path, label: str, missing: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(f"{label}: missing {path}")
        return None
    try:
        return load_json(path)
    except Exception as exc:  # pragma: no cover - defensive, surfaced to caller
        raise ArtifactLoadError(f"Failed to load {label} from {path}: {exc}") from exc


def load_required_sparse_npz(path: Path, label: str, missing: list[str]) -> sparse.csr_matrix | None:
    if not path.exists():
        missing.append(f"{label}: missing {path}")
        return None
    try:
        return load_sparse_npz(path)
    except Exception as exc:  # pragma: no cover - defensive, surfaced to caller
        raise ArtifactLoadError(f"Failed to load {label} from {path}: {exc}") from exc


def load_required_dense_npz(path: Path, label: str, missing: list[str]) -> np.ndarray | None:
    if not path.exists():
        missing.append(f"{label}: missing {path}")
        return None
    try:
        return load_dense_npz(path)
    except Exception as exc:  # pragma: no cover - defensive, surfaced to caller
        raise ArtifactLoadError(f"Failed to load {label} from {path}: {exc}") from exc


def validate_required_columns(table: pd.DataFrame, label: str, required: tuple[str, ...], problems: list[str]) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        problems.append(f"{label}: missing columns {', '.join(missing)}")


def raise_artifact_issues(root: Path, issues: list[str]) -> None:
    if not issues:
        return
    message = "\n".join(f"- {issue}" for issue in issues)
    raise ArtifactLoadError(f"Incomplete or incompatible artifacts under {root}:\n{message}")


def load_artifacts(root: Path) -> LoadedArtifacts:
    business_dir = root / "business_repr"
    manual_dir = root / "user_manual_repr"
    deep_dir = root / "user_deep_repr"
    missing: list[str] = []

    business_ids_df = load_required_tabular(business_dir, "business_ids", "business_ids", missing)
    business_ids = business_ids_df["business_id"] if business_ids_df is not None and "business_id" in business_ids_df.columns else None
    business_full = load_required_sparse_npz(business_dir / "business_full_features.npz", "business_full_features", missing)
    business_deep = load_required_dense_npz(deep_dir / "business_deep_features.npz", "business_deep_features", missing)
    business_table = load_required_tabular(business_dir, "clean_business_table", "business_table", missing)
    business_summary = load_required_json(business_dir / "business_representation_summary.json", "business_representation_summary", missing)
    business_block_summary = load_required_tabular(business_dir, "business_block_summary", "business_block_summary", missing)

    user_manual_ids_df = load_required_tabular(manual_dir, "user_ids", "user_manual_ids", missing)
    user_manual_ids = user_manual_ids_df["user_id"] if user_manual_ids_df is not None and "user_id" in user_manual_ids_df.columns else None
    user_manual_profile = load_required_sparse_npz(manual_dir / "user_profile_features.npz", "user_manual_profile", missing)
    user_manual_table = load_required_tabular(manual_dir, "clean_user_table", "user_manual_table", missing)
    user_manual_summary = load_required_json(manual_dir / "user_profile_summary.json", "user_profile_summary", missing)

    user_deep_ids_df = load_required_tabular(deep_dir, "user_deep_ids", "user_deep_ids", missing)
    user_deep_ids = user_deep_ids_df["user_id"] if user_deep_ids_df is not None and "user_id" in user_deep_ids_df.columns else None
    user_deep = load_required_dense_npz(deep_dir / "user_deep_features.npz", "user_deep_features", missing)
    user_deep_table = load_required_tabular(deep_dir, "user_deep_clean_table", "user_deep_table", missing)
    user_deep_summary = load_required_json(deep_dir / "user_deep_summary.json", "user_deep_summary", missing)

    if business_ids_df is not None and "business_id" not in business_ids_df.columns:
        missing.append("business_ids: missing column business_id")
    if user_manual_ids_df is not None and "user_id" not in user_manual_ids_df.columns:
        missing.append("user_manual_ids: missing column user_id")
    if user_deep_ids_df is not None and "user_id" not in user_deep_ids_df.columns:
        missing.append("user_deep_ids: missing column user_id")

    raise_artifact_issues(
        root,
        missing,
    )

    schema_problems: list[str] = []
    validate_required_columns(business_table, "business_table", REQUIRED_COLUMNS["business_table"], schema_problems)
    validate_required_columns(business_block_summary, "business_block_summary", REQUIRED_COLUMNS["business_block_summary"], schema_problems)
    validate_required_columns(user_manual_table, "user_manual_table", REQUIRED_COLUMNS["user_manual_table"], schema_problems)
    validate_required_columns(user_deep_table, "user_deep_table", REQUIRED_COLUMNS["user_deep_table"], schema_problems)
    raise_artifact_issues(root, schema_problems)

    return LoadedArtifacts(
        business_ids=business_ids.reset_index(drop=True),
        business_full=business_full,
        business_deep=business_deep,
        business_table=business_table,
        business_summary=business_summary,
        business_block_summary=business_block_summary,
        user_manual_ids=user_manual_ids.reset_index(drop=True),
        user_manual_profile=user_manual_profile,
        user_manual_table=user_manual_table,
        user_manual_summary=user_manual_summary,
        user_deep_ids=user_deep_ids.reset_index(drop=True),
        user_deep=user_deep,
        user_deep_table=user_deep_table,
        user_deep_summary=user_deep_summary,
    )


def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def l2_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def dense_health(matrix: np.ndarray, sample_size: int = 50000) -> dict[str, Any]:
    rng = seeded_rng()
    sample_idx = rng.choice(matrix.shape[0], size=min(sample_size, matrix.shape[0]), replace=False)
    sample = matrix[sample_idx]
    norms = np.linalg.norm(sample, axis=1)
    sample_norm = l2_normalize_dense(sample)
    pair_idx = rng.choice(sample_norm.shape[0], size=min(2500, sample_norm.shape[0]), replace=False)
    pair_sample = sample_norm[pair_idx]
    pairwise = pair_sample @ pair_sample.T
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    pca = PCA(n_components=min(8, matrix.shape[1]), random_state=SEED)
    pca.fit(sample)
    dim_var = sample.var(axis=0)
    return {
        "n_rows": int(matrix.shape[0]),
        "n_dims": int(matrix.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_p05": float(np.quantile(norms, 0.05)),
        "norm_p50": float(np.quantile(norms, 0.50)),
        "norm_p95": float(np.quantile(norms, 0.95)),
        "zero_norm_count": int((np.linalg.norm(matrix, axis=1) < 1e-8).sum()),
        "dead_dimension_count": int((dim_var < 1e-8).sum()),
        "random_pair_cosine_mean": float(upper.mean()) if len(upper) else 0.0,
        "random_pair_cosine_std": float(upper.std()) if len(upper) else 0.0,
        "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_.tolist()],
    }


def sparse_health(matrix: sparse.csr_matrix, sample_size: int = 50000) -> dict[str, Any]:
    rng = seeded_rng()
    sample_idx = rng.choice(matrix.shape[0], size=min(sample_size, matrix.shape[0]), replace=False)
    sample = matrix[sample_idx]
    row_norms = np.sqrt(sample.multiply(sample).sum(axis=1)).A1
    nnz = np.diff(sample.indptr)
    return {
        "n_rows": int(matrix.shape[0]),
        "n_dims": int(matrix.shape[1]),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "row_nnz_mean": float(nnz.mean()),
        "row_nnz_p50": float(np.quantile(nnz, 0.50)),
        "row_nnz_p95": float(np.quantile(nnz, 0.95)),
        "norm_mean": float(row_norms.mean()),
        "norm_std": float(row_norms.std()),
        "norm_p05": float(np.quantile(row_norms, 0.05)),
        "norm_p50": float(np.quantile(row_norms, 0.50)),
        "norm_p95": float(np.quantile(row_norms, 0.95)),
    }


def split_tokens(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in text.split("|") if token.strip()]


def assign_macro_category(category_text: Any) -> str:
    tokens = split_tokens(category_text)
    for macro in MACRO_CATEGORY_PRIORITY:
        if macro in tokens:
            return macro
    return tokens[0] if tokens else "Other"


def add_business_semantics(business_table: pd.DataFrame) -> pd.DataFrame:
    table = business_table.copy()
    table["macro_category"] = table["kept_category_tokens"].apply(assign_macro_category)
    table["primary_city"] = table["city_bucket"].fillna("Unknown").astype(str)
    return table


def styled_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    display_df = df.copy()
    for col in float_cols or []:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: ", ".join(f"{float(v):.4f}" for v in x) if isinstance(x, (list, tuple, np.ndarray))
                else (f"{x:.4f}" if pd.notna(x) else "")
            )
    for col in display_df.columns:
        if display_df[col].map(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            display_df[col] = display_df[col].map(lambda x: ", ".join(str(v) for v in x) if isinstance(x, (list, tuple, np.ndarray)) else x)
    return display_df.to_html(index=False, classes="report-table", border=0, escape=False)


def section_intro_html(text: str) -> str:
    return f'<p class="section-lead">{html.escape(text)}</p>'


def explainer_html(*, measures: str, read: str, good_bad: str, warning: str) -> str:
    return (
        '<div class="explain">'
        '<div class="explain-grid">'
        f'<div class="explain-item"><div class="explain-label">Que mide</div><div>{html.escape(measures)}</div></div>'
        f'<div class="explain-item"><div class="explain-label">Como leerlo</div><div>{html.escape(read)}</div></div>'
        f'<div class="explain-item"><div class="explain-label">Senales buenas/malas</div><div>{html.escape(good_bad)}</div></div>'
        f'<div class="explain-item"><div class="explain-label">Advertencia</div><div>{html.escape(warning)}</div></div>'
        '</div>'
        '</div>'
    )


def inject_html_snippets(html_report: str, snippets: list[tuple[str, str]]) -> str:
    rendered = html_report
    for anchor, snippet in snippets:
        if anchor in rendered and snippet:
            rendered = rendered.replace(anchor, anchor + snippet, 1)
    return rendered


def heatmap_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    values = df.to_numpy(dtype=float)
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    span = max(v_max - v_min, 1e-8)
    rows = ['<table class="report-table heatmap"><thead><tr><th></th>']
    rows.extend(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    rows.append("</tr></thead><tbody>")
    for row_name in df.index:
        rows.append(f"<tr><th>{html.escape(str(row_name))}</th>")
        for col in df.columns:
            value = float(df.loc[row_name, col])
            alpha = (value - v_min) / span
            color = f"rgba(15, 76, 92, {0.15 + 0.75 * alpha:.3f})"
            rows.append(f'<td style="background:{color}; color:#fff;">{value:.3f}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def svg_grouped_bar(df: pd.DataFrame, category_col: str, value_cols: list[str], width: int = 820, height: int = 340) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    categories = df[category_col].astype(str).tolist()
    values = df[value_cols].to_numpy(dtype=float)
    max_value = max(float(np.nanmax(values)), 1e-8)
    left, bottom, top, right = 60, 40, 20, 20
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, len(categories))
    bar_w = group_w / max(1, len(value_cols) + 1)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>')
    for index, category in enumerate(categories):
        group_x = left + index * group_w
        for j, value_col in enumerate(value_cols):
            value = safe_float(df.iloc[index][value_col])
            bar_h = 0.0 if math.isnan(value) else (value / max_value) * plot_h
            x = group_x + (j + 0.5) * bar_w
            y = top + plot_h - bar_h
            color = PALETTE[j % len(PALETTE)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.75:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{group_x + group_w / 2:.1f}" y="{height - 10}" text-anchor="middle" font-size="11">{html.escape(category)}</text>')
    for step in range(5):
        value = max_value * step / 4
        y = top + plot_h - plot_h * step / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>')
    legend_x = left + 10
    legend_y = 18
    for j, value_col in enumerate(value_cols):
        color = PALETTE[j % len(PALETTE)]
        parts.append(f'<rect x="{legend_x + j * 180}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18 + j * 180}" y="{legend_y}" font-size="11">{html.escape(value_col)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, width: int = 820, height: int = 420) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    left, bottom, top, right = 60, 40, 20, 20
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_vals = df[x_col].to_numpy(dtype=float)
    y_vals = df[y_col].to_numpy(dtype=float)
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    labels = df[color_col].astype(str).tolist()
    unique_labels = list(dict.fromkeys(labels))
    color_map = {label: PALETTE[index % len(PALETTE)] for index, label in enumerate(unique_labels)}
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#cbd5e1"/>')
    for row in df.itertuples(index=False):
        x = left + (getattr(row, x_col) - x_min) / x_span * plot_w
        y = top + plot_h - (getattr(row, y_col) - y_min) / y_span * plot_h
        color = color_map[str(getattr(row, color_col))]
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}" opacity="0.75"/>')
    legend_x = left
    legend_y = height - 8
    for index, label in enumerate(unique_labels[:10]):
        color = color_map[label]
        parts.append(f'<rect x="{legend_x + index * 78}" y="{legend_y - 10}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 14 + index * 78}" y="{legend_y}" font-size="10">{html.escape(label[:10])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def coverage_section(artifacts: LoadedArtifacts, users_df: pd.DataFrame) -> dict[str, Any]:
    deep_table = artifacts.user_deep_table.copy()
    source_counts = deep_table["embedding_source"].value_counts(dropna=False).rename_axis("embedding_source").reset_index(name="count")
    band_counts = deep_table["history_band"].value_counts(dropna=False).rename_axis("history_band").reset_index(name="count")
    source_band = (
        deep_table.groupby(["history_band", "embedding_source"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["history_band", "embedding_source"])
    )
    users_set = set(users_df["user_id"].astype(str))
    deep_set = set(artifacts.user_deep_ids.astype(str))
    return {
        "kpis": [
            {"label": "Negocios con embedding completo", "value": f"{len(artifacts.business_ids):,}"},
            {"label": "Usuarios con embedding manual", "value": f"{len(artifacts.user_manual_ids):,}"},
            {"label": "Usuarios con embedding profundo", "value": f"{len(artifacts.user_deep_ids):,}"},
            {"label": "Usuarios del snapshot con embedding profundo", "value": format_pct(len(users_set & deep_set) / max(1, len(users_set)))},
        ],
        "source_counts": source_counts,
        "band_counts": band_counts,
        "source_band": source_band,
        "business_blocks": artifacts.business_block_summary.sort_values("start_index"),
        "health": {
            "business_full": sparse_health(artifacts.business_full),
            "business_deep": dense_health(artifacts.business_deep),
            "user_manual_profile": sparse_health(artifacts.user_manual_profile),
            "user_deep": dense_health(artifacts.user_deep),
        },
        "manual_summary": artifacts.user_manual_summary,
        "deep_summary": artifacts.user_deep_summary,
        "business_summary": artifacts.business_summary,
    }


def build_interaction_frame(train_reviews: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_reviews(train_reviews)
    return canonical[["user", "item", "rating", "timestamp"]].dropna(subset=["user", "item", "rating"]).copy()


def sample_frame(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)


def compute_dense_pair_features(user_matrix: np.ndarray, item_matrix: np.ndarray, user_idx: np.ndarray, item_idx: np.ndarray, history_log1p: np.ndarray) -> np.ndarray:
    u = user_matrix[user_idx]
    v = item_matrix[item_idx]
    dot = np.einsum("ij,ij->i", u, v)
    user_norm = np.linalg.norm(u, axis=1)
    item_norm = np.linalg.norm(v, axis=1)
    cosine = dot / np.maximum(user_norm * item_norm, 1e-8)
    return np.column_stack([cosine, dot, user_norm, item_norm, np.abs(user_norm - item_norm), history_log1p]).astype(np.float32)


def compute_sparse_pair_features(user_matrix: sparse.csr_matrix, item_matrix: sparse.csr_matrix, user_idx: np.ndarray, item_idx: np.ndarray, history_log1p: np.ndarray) -> np.ndarray:
    u = user_matrix[user_idx]
    v = item_matrix[item_idx]
    dot = np.asarray(u.multiply(v).sum(axis=1)).ravel()
    user_norm = np.sqrt(np.asarray(u.multiply(u).sum(axis=1)).ravel())
    item_norm = np.sqrt(np.asarray(v.multiply(v).sum(axis=1)).ravel())
    cosine = dot / np.maximum(user_norm * item_norm, 1e-8)
    return np.column_stack([cosine, dot, user_norm, item_norm, np.abs(user_norm - item_norm), history_log1p]).astype(np.float32)


def compute_preference_auc(eval_df: pd.DataFrame) -> tuple[float, int]:
    aucs: list[float] = []
    usable = 0
    for _, group in eval_df.groupby("user", sort=False):
        pos = group.loc[group["rating"] >= 4.0, "pred"].to_numpy(dtype=np.float32)[:20]
        neg = group.loc[group["rating"] <= 2.0, "pred"].to_numpy(dtype=np.float32)[:20]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        aucs.append(float((diff > 0).mean() + 0.5 * (diff == 0).mean()))
        usable += 1
    return (float(np.mean(aucs)) if aucs else float("nan"), usable)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_utility_metrics(artifacts: LoadedArtifacts, train_reviews: pd.DataFrame) -> dict[str, Any]:
    interactions = build_interaction_frame(train_reviews)
    train_split, val_split = temporal_train_validation_split(interactions, val_size=0.2, timestamp_col="timestamp")
    business_index = pd.Series(np.arange(len(artifacts.business_ids), dtype=np.int32), index=artifacts.business_ids.to_numpy())
    manual_user_index = pd.Series(np.arange(len(artifacts.user_manual_ids), dtype=np.int32), index=artifacts.user_manual_ids.to_numpy())
    deep_user_index = pd.Series(np.arange(len(artifacts.user_deep_ids), dtype=np.int32), index=artifacts.user_deep_ids.to_numpy())
    history_lookup = pd.Series(artifacts.user_deep_table["history_count_train"].to_numpy(dtype=np.float32), index=artifacts.user_deep_table["user_id"].to_numpy())
    band_lookup = pd.Series(artifacts.user_deep_table["history_band"].to_numpy(), index=artifacts.user_deep_table["user_id"].to_numpy())

    train_eval = train_split[train_split["user"].isin(deep_user_index.index) & train_split["item"].isin(business_index.index)].copy()
    val_eval = val_split[val_split["user"].isin(deep_user_index.index) & val_split["item"].isin(business_index.index)].copy()
    train_eval = sample_frame(train_eval, 220000)
    val_eval = sample_frame(val_eval, 120000)
    if train_eval.empty or val_eval.empty:
        raise ArtifactLoadError(
            "Temporal evaluation split is empty after filtering to exported embeddings. "
            "The report cannot compare honest validation with post-export diagnostics."
        )

    for frame in (train_eval, val_eval):
        frame["business_idx"] = frame["item"].map(business_index).astype(np.int32)
        frame["manual_user_idx"] = frame["user"].map(manual_user_index).astype(np.int32)
        frame["deep_user_idx"] = frame["user"].map(deep_user_index).astype(np.int32)
        frame["history_log1p"] = np.log1p(frame["user"].map(history_lookup).fillna(0.0).to_numpy(dtype=np.float32))
    val_eval["history_band"] = val_eval["user"].map(band_lookup).fillna("Unknown")

    manual_x_train = compute_sparse_pair_features(
        artifacts.user_manual_profile,
        artifacts.business_full,
        train_eval["manual_user_idx"].to_numpy(dtype=np.int32),
        train_eval["business_idx"].to_numpy(dtype=np.int32),
        train_eval["history_log1p"].to_numpy(dtype=np.float32),
    )
    manual_x_val = compute_sparse_pair_features(
        artifacts.user_manual_profile,
        artifacts.business_full,
        val_eval["manual_user_idx"].to_numpy(dtype=np.int32),
        val_eval["business_idx"].to_numpy(dtype=np.int32),
        val_eval["history_log1p"].to_numpy(dtype=np.float32),
    )
    deep_x_train = compute_dense_pair_features(
        artifacts.user_deep,
        artifacts.business_deep,
        train_eval["deep_user_idx"].to_numpy(dtype=np.int32),
        train_eval["business_idx"].to_numpy(dtype=np.int32),
        train_eval["history_log1p"].to_numpy(dtype=np.float32),
    )
    deep_x_val = compute_dense_pair_features(
        artifacts.user_deep,
        artifacts.business_deep,
        val_eval["deep_user_idx"].to_numpy(dtype=np.int32),
        val_eval["business_idx"].to_numpy(dtype=np.int32),
        val_eval["history_log1p"].to_numpy(dtype=np.float32),
    )
    y_train = train_eval["rating"].to_numpy(dtype=np.float32)
    y_val = val_eval["rating"].to_numpy(dtype=np.float32)

    manual_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    manual_model.fit(manual_x_train, y_train)
    manual_pred = manual_model.predict(manual_x_val)
    deep_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    deep_model.fit(deep_x_train, y_train)
    deep_pred = deep_model.predict(deep_x_val)

    honest_rows = [
        {
            "space": "deep_user_encoder original",
            "evaluation": "validacion temporal interna del entrenamiento",
            "mae": float(artifacts.user_deep_summary["best_val_mae"]),
            "rmse": float(artifacts.user_deep_summary["best_val_rmse"]),
        }
    ]
    diagnostic_rows = [
        {
            "space": "manual_profile + business_full",
            "evaluation": "diagnostico post-export con scorer lineal sobre el split temporal",
            "mae": float(mean_absolute_error(y_val, manual_pred)),
            "rmse": rmse(y_val, manual_pred),
        },
        {
            "space": "user_deep + business_deep",
            "evaluation": "diagnostico post-export con scorer lineal sobre el split temporal",
            "mae": float(mean_absolute_error(y_val, deep_pred)),
            "rmse": rmse(y_val, deep_pred),
        },
    ]
    manual_eval = val_eval[["user", "rating", "history_band"]].copy()
    manual_eval["pred"] = manual_pred
    deep_eval = val_eval[["user", "rating", "history_band"]].copy()
    deep_eval["pred"] = deep_pred
    manual_auc, manual_auc_users = compute_preference_auc(manual_eval)
    deep_auc, deep_auc_users = compute_preference_auc(deep_eval)
    band_rows: list[dict[str, Any]] = []
    for history_band in ["0", "1", "2-5", "6-20", ">20"]:
        mask = val_eval["history_band"] == history_band
        if int(mask.sum()) == 0:
            continue
        band_rows.append(
            {
                "history_band": history_band,
                "manual_mae": float(mean_absolute_error(y_val[mask], manual_pred[mask])),
                "deep_mae": float(mean_absolute_error(y_val[mask], deep_pred[mask])),
                "n_samples": int(mask.sum()),
            }
        )
    honest_table = pd.DataFrame(honest_rows)
    diagnostic_table = pd.DataFrame(diagnostic_rows)
    return {
        "honest_table": honest_table,
        "diagnostic_table": diagnostic_table,
        "preference_table": pd.DataFrame(
            [
                {"space": "manual_profile + business_full", "pairwise_auc": manual_auc, "n_users": manual_auc_users},
                {"space": "user_deep + business_deep", "pairwise_auc": deep_auc, "n_users": deep_auc_users},
            ]
        ),
        "band_table": pd.DataFrame(band_rows),
        "honest_summary": {
            "deep_mae": float(artifacts.user_deep_summary["best_val_mae"]),
            "deep_rmse": float(artifacts.user_deep_summary["best_val_rmse"]),
        },
        "diagnostic_summary": {
            "manual_mae": float(mean_absolute_error(y_val, manual_pred)),
            "manual_rmse": rmse(y_val, manual_pred),
            "deep_mae": float(mean_absolute_error(y_val, deep_pred)),
            "deep_rmse": rmse(y_val, deep_pred),
        },
    }


def pick_top_categories(business_table: pd.DataFrame, min_count: int = 250, top_n: int = 10) -> list[str]:
    counts = business_table["macro_category"].value_counts()
    return counts[counts >= min_count].head(top_n).index.tolist()


def grouped_sample_indices(labels: pd.Series, max_per_group: int) -> np.ndarray:
    rng = seeded_rng()
    all_indices: list[np.ndarray] = []
    values = labels.astype(str)
    for label in values.dropna().unique():
        idx = np.flatnonzero(values.to_numpy() == label)
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        all_indices.append(np.asarray(idx, dtype=np.int32))
    return np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)


def compute_knn_metrics(normalized_matrix: Any, categories: np.ndarray, cities: np.ndarray, anchor_idx: np.ndarray, top_k: int, is_sparse: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor in anchor_idx:
        sim = normalized_matrix[anchor].dot(normalized_matrix.T).toarray().ravel() if is_sparse else normalized_matrix[anchor] @ normalized_matrix.T
        sim[anchor] = -np.inf
        top_idx = np.argpartition(sim, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sim[top_idx])[::-1]]
        neighbor_cats = categories[top_idx]
        neighbor_cities = cities[top_idx]
        cat_counter = Counter(neighbor_cats.tolist())
        entropy = 0.0
        for value in cat_counter.values():
            prob = value / max(1, len(neighbor_cats))
            entropy -= prob * math.log(prob + 1e-12)
        rows.append(
            {
                "anchor_category": categories[anchor],
                "same_category_ratio": float(np.mean(neighbor_cats == categories[anchor])),
                "same_city_ratio": float(np.mean(neighbor_cities == cities[anchor])),
                "category_entropy": float(entropy),
            }
        )
    return pd.DataFrame(rows)


def build_business_semantic_metrics(artifacts: LoadedArtifacts, top_k: int, sample_business_pca: int, business_anchor_per_category: int) -> dict[str, Any]:
    business_table = add_business_semantics(artifacts.business_table)
    top_categories = pick_top_categories(business_table)
    category_mask = business_table["macro_category"].isin(top_categories)
    anchor_idx = grouped_sample_indices(business_table.loc[category_mask, "macro_category"], business_anchor_per_category)
    anchor_idx = business_table.loc[category_mask].iloc[anchor_idx].index.to_numpy(dtype=np.int32)
    dense_norm = l2_normalize_dense(artifacts.business_deep)
    sparse_norm = normalize(artifacts.business_full, norm="l2", axis=1, copy=True)
    category_values = business_table["macro_category"].to_numpy()
    city_values = business_table["primary_city"].to_numpy()
    deep_knn = compute_knn_metrics(dense_norm, category_values, city_values, anchor_idx, top_k=top_k, is_sparse=False)
    full_knn = compute_knn_metrics(sparse_norm, category_values, city_values, anchor_idx, top_k=top_k, is_sparse=True)
    deep_knn["space"] = "business_deep"
    full_knn["space"] = "business_full"
    coherence = pd.concat([deep_knn, full_knn], ignore_index=True)
    category_summary = coherence.groupby(["space", "anchor_category"], as_index=False)[["same_category_ratio", "same_city_ratio", "category_entropy"]].mean().sort_values(["space", "same_category_ratio"], ascending=[True, False])
    overall_summary = coherence.groupby("space", as_index=False)[["same_category_ratio", "same_city_ratio", "category_entropy"]].mean().sort_values("same_category_ratio", ascending=False)

    rng = seeded_rng()
    pca_sample_idx = rng.choice(len(business_table), size=min(sample_business_pca, len(business_table)), replace=False)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(dense_norm[pca_sample_idx])
    scatter_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "category": business_table.iloc[pca_sample_idx]["macro_category"].to_numpy()})

    centroid_rows = []
    for category in top_categories:
        idx = np.flatnonzero(category_values == category)[:500]
        if len(idx):
            centroid_rows.append({"category": category, "vector": dense_norm[idx].mean(axis=0)})
    centroid_categories = [row["category"] for row in centroid_rows]
    centroid_matrix = l2_normalize_dense(np.vstack([row["vector"] for row in centroid_rows]))
    centroid_df = pd.DataFrame(centroid_matrix @ centroid_matrix.T, index=centroid_categories, columns=centroid_categories)
    return {"overall_summary": overall_summary, "category_summary": category_summary, "scatter": scatter_df, "centroid_similarity": centroid_df}


def compute_user_similarity_consistency(artifacts: LoadedArtifacts, sample_size: int) -> dict[str, Any]:
    eligible_idx = np.flatnonzero(artifacts.user_deep_table["history_count_train"].fillna(0).to_numpy() >= 2)
    if len(eligible_idx) == 0:
        return {"pairwise_corr": float("nan"), "neighbor_overlap_mean": float("nan"), "n_users": 0}
    rng = seeded_rng()
    chosen = rng.choice(eligible_idx, size=min(sample_size, len(eligible_idx)), replace=False)
    deep_sample = l2_normalize_dense(artifacts.user_deep[chosen])
    manual_sample = normalize(artifacts.user_manual_profile[chosen].toarray().astype(np.float32), norm="l2", axis=1, copy=False)
    deep_dist = 1.0 - (deep_sample @ deep_sample.T)
    manual_dist = 1.0 - (manual_sample @ manual_sample.T)
    upper = np.triu_indices_from(deep_dist, k=1)
    corr = float(np.corrcoef(deep_dist[upper], manual_dist[upper])[0, 1])
    k = 10
    deep_neighbors = np.argsort(deep_dist, axis=1)[:, 1 : k + 1]
    manual_neighbors = np.argsort(manual_dist, axis=1)[:, 1 : k + 1]
    overlaps = [len(set(deep_neighbors[i].tolist()) & set(manual_neighbors[i].tolist())) / k for i in range(len(chosen))]
    return {"pairwise_corr": corr, "neighbor_overlap_mean": float(np.mean(overlaps)), "n_users": int(len(chosen))}


def build_macro_history_lookup(train_reviews: pd.DataFrame, business_table: pd.DataFrame) -> pd.DataFrame:
    macro_map = business_table[["business_id", "macro_category"]].drop_duplicates()
    interactions = canonicalize_reviews(train_reviews).merge(macro_map, left_on="item", right_on="business_id", how="left")
    interactions["macro_category"] = interactions["macro_category"].fillna("Other")
    grouped = interactions.groupby(["user", "macro_category"], as_index=False).size().sort_values(["user", "size", "macro_category"], ascending=[True, False, True])
    dominant = grouped.drop_duplicates("user").rename(columns={"user": "user_id", "macro_category": "dominant_macro_category", "size": "dominant_macro_count"})
    return dominant[["user_id", "dominant_macro_category", "dominant_macro_count"]]


def _label_cluster(row: pd.Series) -> str:
    history = safe_float(row["history_median"])
    social = safe_float(row["social_capital_mean_z"])
    macro = str(row.get("dominant_macro", "Mixed"))
    activity = "muy activos" if history >= 8 else "activos" if history >= 3 else "ligeros"
    social_label = "alta proyeccion social" if social >= 2.0 else "perfil discreto"
    return f"{macro} | {activity} | {social_label}"


def _cluster_one_population(user_table: pd.DataFrame, embedding_matrix: np.ndarray, indices: np.ndarray, dominant_category_df: pd.DataFrame, cluster_count: int, name: str) -> dict[str, Any]:
    if len(indices) == 0:
        return {"name": name, "cluster_profiles": pd.DataFrame(), "scatter": pd.DataFrame(), "silhouette": float("nan"), "n_users": 0}
    subset = user_table.iloc[indices].copy()
    reduced = PCA(n_components=min(32, embedding_matrix.shape[1]), random_state=SEED).fit_transform(l2_normalize_dense(embedding_matrix[indices]))
    model = MiniBatchKMeans(n_clusters=min(cluster_count, len(indices)), random_state=SEED, batch_size=2048, n_init=5)
    labels = model.fit_predict(reduced)
    subset["cluster_id"] = labels
    silhouette = float("nan")
    if len(np.unique(labels)) > 1 and len(subset) > 500:
        sample_size = min(10000, len(subset))
        silhouette = float(silhouette_score(reduced[:sample_size], labels[:sample_size], metric="euclidean"))
    subset = subset.merge(dominant_category_df, on="user_id", how="left")
    subset["dominant_macro_category"] = subset["dominant_macro_category"].fillna("No history")
    subset["social_capital_z"] = subset[["metadata__fans_log1p_z", "metadata__useful_log1p_z", "metadata__cool_log1p_z"]].mean(axis=1)
    cluster_profiles = subset.groupby("cluster_id", as_index=False).agg(
        n_users=("user_id", "size"),
        history_median=("history_count_train", "median"),
        tenure_mean=("metadata__tenure_days", "mean"),
        elite_rate=("metadata__elite_any", "mean"),
        fans_mean_z=("metadata__fans_log1p_z", "mean"),
        social_capital_mean_z=("social_capital_z", "mean"),
    ).sort_values("n_users", ascending=False)
    dominant_labels = subset.groupby(["cluster_id", "dominant_macro_category"], as_index=False).size().sort_values(["cluster_id", "size", "dominant_macro_category"], ascending=[True, False, True]).drop_duplicates("cluster_id").rename(columns={"dominant_macro_category": "dominant_macro", "size": "dominant_macro_support"})
    cluster_profiles = cluster_profiles.merge(dominant_labels[["cluster_id", "dominant_macro", "dominant_macro_support"]], on="cluster_id", how="left")
    cluster_profiles["cluster_label"] = cluster_profiles.apply(_label_cluster, axis=1)
    scatter = pd.DataFrame({"x": reduced[: min(6000, len(subset)), 0], "y": reduced[: min(6000, len(subset)), 1], "cluster": subset.iloc[: min(6000, len(subset))]["cluster_id"].astype(str).to_numpy()})
    return {"name": name, "cluster_profiles": cluster_profiles, "scatter": scatter, "silhouette": silhouette, "n_users": int(len(subset))}


def build_cluster_profiles(user_table: pd.DataFrame, embedding_matrix: np.ndarray, dominant_category_df: pd.DataFrame, history_target: int, cold_target: int) -> dict[str, Any]:
    rng = seeded_rng()
    history_mask = (user_table["embedding_source"] == "history") & user_table["history_band"].isin(["2-5", "6-20", ">20"])
    cold_mask = user_table["embedding_source"].isin(["metadata_only", "default_only"])
    history_idx = np.flatnonzero(history_mask.to_numpy())
    cold_idx = np.flatnonzero(cold_mask.to_numpy())
    if len(history_idx):
        history_idx = rng.choice(history_idx, size=min(history_target, len(history_idx)), replace=False)
    if len(cold_idx):
        cold_idx = rng.choice(cold_idx, size=min(cold_target, len(cold_idx)), replace=False)
    return {
        "historical": _cluster_one_population(user_table, embedding_matrix, history_idx, dominant_category_df, 12, "historical"),
        "cold_start": _cluster_one_population(user_table, embedding_matrix, cold_idx, dominant_category_df, 8, "cold_start"),
    }


def parse_friends(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    value = str(text).strip()
    if not value or value == "None":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def bucket_from_series(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    clipped = values.fillna(values.median() if not values.dropna().empty else 0.0)
    try:
        return pd.cut(clipped, bins=bins, labels=labels, include_lowest=True).astype(str)
    except ValueError:
        quantiles = min(len(labels), max(1, int(clipped.nunique())))
        effective_labels = labels[:quantiles]
        return pd.qcut(clipped.rank(method="first"), q=quantiles, labels=effective_labels, duplicates="drop").astype(str)


def build_friend_similarity_analysis(artifacts: LoadedArtifacts, users_df: pd.DataFrame, friend_anchor_target: int, friend_pairs_per_anchor: int) -> dict[str, Any]:
    deep_table = artifacts.user_deep_table.copy().merge(users_df[["user_id", "friends"]], on="user_id", how="left")
    valid_user_ids = set(users_df["user_id"].astype(str)) & set(artifacts.user_deep_ids.astype(str))
    deep_table = deep_table[deep_table["user_id"].isin(valid_user_ids)].copy()
    degree_counts: list[int] = []
    directed_valid = 0
    undirected_est = 0
    anchor_candidates: list[str] = []
    for row in deep_table[["user_id", "friends"]].itertuples(index=False):
        valid_friends = [friend for friend in parse_friends(row.friends) if friend in valid_user_ids and friend != row.user_id]
        degree = len(valid_friends)
        degree_counts.append(degree)
        if degree:
            anchor_candidates.append(row.user_id)
            directed_valid += degree
            undirected_est += sum(1 for friend in valid_friends if row.user_id < friend)

    profile = deep_table[["user_id", "history_band", "embedding_source", "metadata__tenure_days", "metadata__fans_log1p_z"]].copy()
    profile["activity_bucket"] = profile["history_band"].astype(str).replace({"6-20": "6+", ">20": "6+"})
    tenure_bins = list(np.quantile(profile["metadata__tenure_days"].fillna(0.0), [0.0, 0.25, 0.50, 0.75, 1.0]))
    tenure_bins = [tenure_bins[0] - 1e-6, tenure_bins[1], tenure_bins[2], tenure_bins[3], tenure_bins[4] + 1e-6]
    fans_bins = list(np.quantile(profile["metadata__fans_log1p_z"].fillna(0.0), [0.0, 0.25, 0.50, 0.75, 1.0]))
    fans_bins = [fans_bins[0] - 1e-6, fans_bins[1], fans_bins[2], fans_bins[3], fans_bins[4] + 1e-6]
    profile["tenure_bucket"] = bucket_from_series(profile["metadata__tenure_days"], tenure_bins, ["q1", "q2", "q3", "q4"])
    profile["fans_bucket"] = bucket_from_series(profile["metadata__fans_log1p_z"], fans_bins, ["q1", "q2", "q3", "q4"])
    profile["stratum"] = profile["embedding_source"].astype(str) + "|" + profile["activity_bucket"].astype(str) + "|" + profile["tenure_bucket"].astype(str) + "|" + profile["fans_bucket"].astype(str)
    profile_lookup = profile.set_index("user_id")
    stratum_pools: dict[str, list[str]] = defaultdict(list)
    for row in profile[["user_id", "stratum"]].itertuples(index=False):
        stratum_pools[row.stratum].append(row.user_id)

    rng = seeded_rng()
    anchor_series = pd.Series(anchor_candidates)
    sampled_anchors = set(anchor_series.sample(n=min(friend_anchor_target, len(anchor_series)), random_state=SEED).tolist()) if len(anchor_series) else set()
    anchor_friend_map: dict[str, list[str]] = {}
    for row in deep_table[["user_id", "friends"]].itertuples(index=False):
        if row.user_id not in sampled_anchors:
            continue
        valid_friends = [friend for friend in parse_friends(row.friends) if friend in valid_user_ids and friend != row.user_id]
        if valid_friends:
            anchor_friend_map[row.user_id] = valid_friends

    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    for anchor, friends in anchor_friend_map.items():
        chosen = friends if len(friends) <= friend_pairs_per_anchor else rng.choice(friends, size=friend_pairs_per_anchor, replace=False).tolist()
        friend_set = set(friends)
        for friend in chosen:
            positives.append((anchor, friend))
            stratum = str(profile_lookup.loc[friend, "stratum"]) if friend in profile_lookup.index else ""
            pool = stratum_pools.get(stratum, [])
            neg = None
            for _ in range(40):
                candidate = pool[int(rng.integers(0, len(pool)))] if pool else None
                if candidate is None or candidate == anchor or candidate in friend_set:
                    continue
                neg = candidate
                break
            if neg is not None:
                negatives.append((anchor, neg))

    pair_count = min(len(positives), len(negatives))
    positives = positives[:pair_count]
    negatives = negatives[:pair_count]
    deep_index = pd.Series(np.arange(len(artifacts.user_deep_ids), dtype=np.int32), index=artifacts.user_deep_ids.to_numpy())
    manual_index = pd.Series(np.arange(len(artifacts.user_manual_ids), dtype=np.int32), index=artifacts.user_manual_ids.to_numpy())
    deep_norm = l2_normalize_dense(artifacts.user_deep)

    def pair_cos_dense(pairs: list[tuple[str, str]]) -> np.ndarray:
        left = np.array([deep_index[u] for u, _ in pairs], dtype=np.int32)
        right = np.array([deep_index[v] for _, v in pairs], dtype=np.int32)
        return np.einsum("ij,ij->i", deep_norm[left], deep_norm[right])

    def pair_cos_manual(pairs: list[tuple[str, str]]) -> np.ndarray:
        ids = sorted({user for pair in pairs for user in pair})
        local = l2_normalize_dense(artifacts.user_manual_profile[[manual_index[user] for user in ids]].toarray().astype(np.float32))
        local_index = {user_id: pos for pos, user_id in enumerate(ids)}
        return np.array([float(np.dot(local[local_index[left]], local[local_index[right]])) for left, right in pairs], dtype=np.float32)

    deep_pos = pair_cos_dense(positives)
    deep_neg = pair_cos_dense(negatives)
    manual_pos = pair_cos_manual(positives)
    manual_neg = pair_cos_manual(negatives)
    pair_frame = pd.DataFrame(
        {
            "anchor": [pair[0] for pair in positives],
            "friend": [pair[1] for pair in positives],
            "matched_non_friend": [pair[1] for pair in negatives],
            "history_band": [profile_lookup.loc[pair[0], "history_band"] for pair in positives],
            "anchor_source": [profile_lookup.loc[pair[0], "embedding_source"] for pair in positives],
            "friend_source": [profile_lookup.loc[pair[1], "embedding_source"] for pair in positives],
            "deep_friend_cos": deep_pos,
            "deep_non_friend_cos": deep_neg,
            "manual_friend_cos": manual_pos,
            "manual_non_friend_cos": manual_neg,
        }
    )
    pair_frame["deep_uplift"] = pair_frame["deep_friend_cos"] - pair_frame["deep_non_friend_cos"]
    pair_frame["manual_uplift"] = pair_frame["manual_friend_cos"] - pair_frame["manual_non_friend_cos"]
    summary_rows = [
        {"space": "user_deep", "friend_mean_cos": float(pair_frame["deep_friend_cos"].mean()), "matched_non_friend_mean_cos": float(pair_frame["deep_non_friend_cos"].mean()), "uplift": float(pair_frame["deep_uplift"].mean()), "paired_win_rate": float((pair_frame["deep_uplift"] > 0).mean())},
        {"space": "user_manual_profile", "friend_mean_cos": float(pair_frame["manual_friend_cos"].mean()), "matched_non_friend_mean_cos": float(pair_frame["manual_non_friend_cos"].mean()), "uplift": float(pair_frame["manual_uplift"].mean()), "paired_win_rate": float((pair_frame["manual_uplift"] > 0).mean())},
    ]
    history_history = pair_frame[(pair_frame["anchor_source"] == "history") & (pair_frame["friend_source"] == "history")]
    primary_rows = []
    if not history_history.empty:
        primary_rows = [
            {"space": "user_deep", "subset": "history-history", "uplift": float(history_history["deep_uplift"].mean()), "paired_win_rate": float((history_history["deep_uplift"] > 0).mean()), "n_pairs": int(len(history_history))},
            {"space": "user_manual_profile", "subset": "history-history", "uplift": float(history_history["manual_uplift"].mean()), "paired_win_rate": float((history_history["manual_uplift"] > 0).mean()), "n_pairs": int(len(history_history))},
        ]
    degree_array = np.array(degree_counts, dtype=np.int32)
    graph_summary = {
        "n_users_in_snapshot": int(len(deep_table)),
        "n_users_with_valid_friend_edges": int((degree_array > 0).sum()),
        "directed_valid_edges": int(directed_valid),
        "undirected_edge_estimate": int(undirected_est),
        "degree_p50": float(np.quantile(degree_array, 0.50)) if len(degree_array) else 0.0,
        "degree_p90": float(np.quantile(degree_array, 0.90)) if len(degree_array) else 0.0,
        "degree_p99": float(np.quantile(degree_array, 0.99)) if len(degree_array) else 0.0,
        "degree_max": int(degree_array.max()) if len(degree_array) else 0,
    }
    return {"graph_summary": graph_summary, "summary_table": pd.DataFrame(summary_rows), "primary_table": pd.DataFrame(primary_rows), "pair_frame": pair_frame}


def build_report_html(*, coverage: dict[str, Any], utility: dict[str, Any], business_metrics: dict[str, Any], user_consistency: dict[str, Any], clusters: dict[str, Any], friends: dict[str, Any], recommendations: dict[str, Any]) -> str:
    template = Template(
        """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Embedding Quality Report</title>
  <style>
    body { font-family: "Segoe UI", Tahoma, sans-serif; margin: 0; color: #0f172a; background: #f8fafc; }
    .layout { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }
    nav { background: linear-gradient(180deg, #0f4c5c 0%, #134e4a 100%); color: #fff; padding: 24px 20px; position: sticky; top: 0; height: 100vh; box-sizing: border-box; }
    nav h1 { font-size: 20px; margin: 0 0 8px 0; }
    nav p { font-size: 13px; line-height: 1.45; color: rgba(255,255,255,0.84); }
    nav a { display: block; color: #fff; text-decoration: none; margin: 10px 0; font-size: 14px; opacity: 0.92; }
    main { padding: 28px 34px 48px; }
    section { margin-bottom: 36px; }
    h2 { margin: 0 0 10px 0; font-size: 28px; }
    h3 { margin: 22px 0 8px 0; font-size: 20px; }
    p { line-height: 1.6; }
    .section-lead { margin: 8px 0 0; color: #334155; max-width: 980px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 16px 0 18px; }
    .card { background: #fff; border-radius: 14px; padding: 16px; box-shadow: 0 6px 20px rgba(15,23,42,0.06); border: 1px solid #e2e8f0; }
    .card .label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #475569; }
    .card .value { font-size: 24px; font-weight: 700; margin-top: 6px; }
    .panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px; box-shadow: 0 8px 26px rgba(15,23,42,0.05); margin-top: 14px; }
    .explain { margin: 10px 0 14px; padding: 12px 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; }
    .explain-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px 14px; }
    .explain-item { font-size: 12.5px; line-height: 1.45; color: #334155; }
    .explain-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #0f4c5c; margin-bottom: 4px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .report-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .report-table th, .report-table td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; }
    .report-table th { background: #f8fafc; }
    .heatmap th, .heatmap td { text-align: center; }
    .chart { width: 100%; height: auto; display: block; }
    ul.flat { padding-left: 18px; }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } nav { position: static; height: auto; } .two-col { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="layout">
  <nav>
    <h1>Embedding Report</h1>
    <p>Diagnóstico integral de utilidad, diversidad semántica, segmentación y homofilia social sobre los embeddings finales de competición.</p>
    <a href="#resumen">Resumen</a><a href="#cobertura">Cobertura y Salud</a><a href="#utilidad">Utilidad</a><a href="#negocio">Negocios</a><a href="#usuarios">Usuarios</a><a href="#amistades">Amistades</a><a href="#recomendaciones">Recomendaciones</a>
  </nav>
  <main>
    <section id="resumen">
      <h2>Resumen Ejecutivo</h2>
      <p>El embedding profundo de usuario tiene cobertura completa del snapshot y una separación clara entre usuarios con historial real y usuarios resueltos por fallback de metadata. En geometría local, el espacio profundo de negocios muestra mejor coherencia semántica que el espacio completo manual, mientras que a nivel social los amigos tienden a estar más cerca que no-amigos emparejados por actividad y metadata.</p>
      <div class="cards">{% for card in coverage.kpis %}<div class="card"><div class="label">{{ card.label }}</div><div class="value">{{ card.value }}</div></div>{% endfor %}<div class="card"><div class="label">MAE temporal oficial deep</div><div class="value">{{ "%.4f"|format(coverage.deep_summary.best_val_mae) }}</div></div><div class="card"><div class="label">Usuarios history-based</div><div class="value">{{ "{:,}".format(coverage.deep_summary.embedding_source_counts.history) }}</div></div><div class="card"><div class="label">Uplift social deep</div><div class="value">{{ "%.4f"|format(friends.summary_table.iloc[0].uplift if not friends.summary_table.empty else 0.0) }}</div></div></div>
    </section>
    <section id="cobertura"><h2>Cobertura y Salud Del Espacio</h2><div class="two-col"><div class="panel"><h3>Fuente del embedding por usuario</h3>{{ coverage.source_counts_html | safe }}</div><div class="panel"><h3>Bandas de historial</h3>{{ coverage.band_counts_html | safe }}</div></div><div class="panel"><h3>Composición `history_band x embedding_source`</h3>{{ coverage.source_band_chart | safe }}</div><div class="two-col"><div class="panel"><h3>Bloques del embedding completo de negocio</h3>{{ coverage.business_blocks_html | safe }}</div><div class="panel"><h3>Salud de las representaciones</h3>{{ coverage.health_html | safe }}</div></div></section>
    <section id="utilidad"><h2>Utilidad Para La Tarea Final</h2><div class="panel"><h3>Comparativa de scorers</h3>{{ utility.summary_html | safe }}</div><div class="two-col"><div class="panel"><h3>AUC de preferencia</h3>{{ utility.preference_html | safe }}</div><div class="panel"><h3>MAE por banda de historial</h3>{{ utility.band_chart | safe }}</div></div></section>
    <section id="negocio"><h2>Embeddings de Negocio</h2><div class="panel"><h3>Coherencia local global</h3>{{ business.overall_html | safe }}</div><div class="two-col"><div class="panel"><h3>Coherencia por categoría</h3>{{ business.category_html | safe }}</div><div class="panel"><h3>Mapa PCA de negocios profundos</h3>{{ business.scatter_svg | safe }}</div></div><div class="panel"><h3>Similitud entre centroides de categorías</h3>{{ business.heatmap_html | safe }}</div></section>
    <section id="usuarios"><h2>Embeddings de Usuario</h2><div class="two-col"><div class="panel"><h3>Consistencia manual vs deep</h3><p><strong>Correlación de distancias:</strong> {{ "%.4f"|format(user_consistency.pairwise_corr) }}</p><p><strong>Solape medio de vecinos top-10:</strong> {{ "%.4f"|format(user_consistency.neighbor_overlap_mean) }}</p><p><strong>Muestra analizada:</strong> {{ "{:,}".format(user_consistency.n_users) }} usuarios con al menos 2 reviews.</p></div><div class="panel"><h3>Clustering histórico</h3><p><strong>Usuarios usados:</strong> {{ "{:,}".format(clusters.historical.n_users) }}</p><p><strong>Silhouette:</strong> {{ "%.4f"|format(clusters.historical.silhouette) if clusters.historical.silhouette == clusters.historical.silhouette else "n/a" }}</p>{{ clusters.historical.table_html | safe }}</div></div><div class="panel"><h3>Mapa PCA de clusters históricos</h3>{{ clusters.historical.scatter_svg | safe }}</div><div class="two-col"><div class="panel"><h3>Clustering cold-start</h3><p><strong>Usuarios usados:</strong> {{ "{:,}".format(clusters.cold_start.n_users) }}</p><p><strong>Silhouette:</strong> {{ "%.4f"|format(clusters.cold_start.silhouette) if clusters.cold_start.silhouette == clusters.cold_start.silhouette else "n/a" }}</p>{{ clusters.cold_start.table_html | safe }}</div><div class="panel"><h3>Mapa PCA cold-start</h3>{{ clusters.cold_start.scatter_svg | safe }}</div></div></section>
    <section id="amistades"><h2>Amistades y Homofilia Social</h2><div class="cards"><div class="card"><div class="label">Usuarios con amigos válidos</div><div class="value">{{ "{:,}".format(friends.graph_summary.n_users_with_valid_friend_edges) }}</div></div><div class="card"><div class="label">Aristas dirigidas válidas</div><div class="value">{{ "{:,}".format(friends.graph_summary.directed_valid_edges) }}</div></div><div class="card"><div class="label">p90 grado social</div><div class="value">{{ "%.0f"|format(friends.graph_summary.degree_p90) }}</div></div><div class="card"><div class="label">máximo grado</div><div class="value">{{ "{:,}".format(friends.graph_summary.degree_max) }}</div></div></div><div class="two-col"><div class="panel"><h3>Resumen friend vs matched non-friend</h3>{{ friends.summary_html | safe }}</div><div class="panel"><h3>Subconjunto history-history</h3>{{ friends.primary_html | safe }}</div></div><div class="panel"><h3>Visualización del uplift social</h3>{{ friends.chart_svg | safe }}</div></section>
    <section id="recomendaciones"><h2>Recomendaciones Para Competición</h2><div class="panel"><ul class="flat"><li>Usar `user_deep + business_deep` como espacio principal para scoring y recuperación semántica, sobre todo en usuarios con `history_band >= 2`.</li><li>Mantener `user_manual_profile + business_full` como baseline robusto y como fuente complementaria de explicabilidad.</li><li>Separar explícitamente `cold-start`, `single-review` e `historical` en el pipeline final.</li><li>Introducir la similitud social como señal auxiliar solo en `history-history` o como regularización.</li><li>El mayor retorno parece estar en mejorar cold-start de usuario y en reforzar el scorer final.</li></ul></div></section>
  </main>
</div>
</body>
</html>
        """
    )
    rendered = template.render(coverage=coverage, utility=utility, business=business_metrics, user_consistency=user_consistency, clusters=clusters, friends=friends)
    notes = [
        ("<h2>Resumen Ejecutivo</h2>", coverage["summary_intro_html"]),
        ("<h2>Cobertura y Salud Del Espacio</h2>", coverage["section_intro_html"]),
        ("<h3>Fuente del embedding por usuario</h3>", coverage["source_counts_note_html"]),
        ("<h3>Bandas de historial</h3>", coverage["band_counts_note_html"]),
        ("<h3>ComposiciÃ³n `history_band x embedding_source`</h3>", coverage["source_band_note_html"]),
        ("<h3>Bloques del embedding completo de negocio</h3>", coverage["business_blocks_note_html"]),
        ("<h3>Salud de las representaciones</h3>", coverage["health_note_html"]),
        ("<h2>Utilidad Para La Tarea Final</h2>", utility["section_intro_html"]),
        ("<h3>Comparativa de scorers</h3>", utility["summary_note_html"]),
        ("<h3>AUC de preferencia</h3>", utility["preference_note_html"]),
        ("<h3>MAE por banda de historial</h3>", utility["band_note_html"]),
        ("<h2>Embeddings de Negocio</h2>", business_metrics["section_intro_html"]),
        ("<h3>Coherencia local global</h3>", business_metrics["overall_note_html"]),
        ("<h3>Coherencia por categorÃ­a</h3>", business_metrics["category_note_html"]),
        ("<h3>Mapa PCA de negocios profundos</h3>", business_metrics["scatter_note_html"]),
        ("<h3>Similitud entre centroides de categorÃ­as</h3>", business_metrics["heatmap_note_html"]),
        ("<h2>Embeddings de Usuario</h2>", user_consistency["section_intro_html"]),
        ("<h3>Consistencia manual vs deep</h3>", user_consistency["note_html"]),
        ("<h3>Clustering histÃ³rico</h3>", clusters["historical"]["note_html"]),
        ("<h3>Mapa PCA de clusters histÃ³ricos</h3>", clusters["historical"]["scatter_note_html"]),
        ("<h3>Clustering cold-start</h3>", clusters["cold_start"]["note_html"]),
        ("<h3>Mapa PCA cold-start</h3>", clusters["cold_start"]["scatter_note_html"]),
        ("<h2>Amistades y Homofilia Social</h2>", friends["section_intro_html"]),
        ("<h3>Resumen friend vs matched non-friend</h3>", friends["summary_note_html"]),
        ("<h3>Subconjunto history-history</h3>", friends["primary_note_html"]),
        ("<h3>VisualizaciÃ³n del uplift social</h3>", friends["chart_note_html"]),
        ("<h2>Recomendaciones Para CompeticiÃ³n</h2>", recommendations["note_html"]),
    ]
    return inject_html_snippets(rendered, notes)


def save_outputs(report_dir: Path, *, coverage: dict[str, Any], utility: dict[str, Any], business_metrics: dict[str, Any], clusters: dict[str, Any], friends: dict[str, Any], html_report: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "embedding_quality_report.html").write_text(html_report, encoding="utf-8")
    utility["summary_table"].to_csv(report_dir / "utility_summary.csv", index=False)
    utility["preference_table"].to_csv(report_dir / "utility_preference_auc.csv", index=False)
    utility["band_table"].to_csv(report_dir / "utility_band_mae.csv", index=False)
    business_metrics["overall_summary"].to_csv(report_dir / "business_coherence_overall.csv", index=False)
    business_metrics["category_summary"].to_csv(report_dir / "business_coherence_by_category.csv", index=False)
    business_metrics["centroid_similarity"].to_csv(report_dir / "business_centroid_similarity.csv")
    clusters["historical"]["cluster_profiles"].to_csv(report_dir / "historical_cluster_profiles.csv", index=False)
    clusters["cold_start"]["cluster_profiles"].to_csv(report_dir / "cold_start_cluster_profiles.csv", index=False)
    friends["summary_table"].to_csv(report_dir / "friend_similarity_summary.csv", index=False)
    friends["primary_table"].to_csv(report_dir / "friend_similarity_primary.csv", index=False)
    friends["pair_frame"].to_csv(report_dir / "friend_similarity_pairs.csv", index=False)
    summary = {
        "deep_best_val_mae": float(coverage["deep_summary"]["best_val_mae"]),
        "deep_best_val_rmse": float(coverage["deep_summary"]["best_val_rmse"]),
        "social_uplift_deep": float(friends["summary_table"].iloc[0]["uplift"]) if not friends["summary_table"].empty else None,
        "historical_cluster_silhouette": float(clusters["historical"]["silhouette"]) if clusters["historical"]["silhouette"] == clusters["historical"]["silhouette"] else None,
        "cold_cluster_silhouette": float(clusters["cold_start"]["silhouette"]) if clusters["cold_start"]["silhouette"] == clusters["cold_start"]["silhouette"] else None,
    }
    (report_dir / "embedding_quality_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_report_html_v2(*, coverage: dict[str, Any], utility: dict[str, Any], business_metrics: dict[str, Any], user_consistency: dict[str, Any], clusters: dict[str, Any], friends: dict[str, Any], recommendations: dict[str, Any]) -> str:
    template = Template(
        """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Embedding Quality Report</title>
  <style>
    body { font-family: "Segoe UI", Tahoma, sans-serif; margin: 0; color: #0f172a; background: #f8fafc; }
    .layout { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }
    nav { background: linear-gradient(180deg, #0f4c5c 0%, #134e4a 100%); color: #fff; padding: 24px 20px; position: sticky; top: 0; height: 100vh; box-sizing: border-box; }
    nav h1 { font-size: 20px; margin: 0 0 8px 0; }
    nav p { font-size: 13px; line-height: 1.45; color: rgba(255,255,255,0.84); }
    nav a { display: block; color: #fff; text-decoration: none; margin: 10px 0; font-size: 14px; opacity: 0.92; }
    main { padding: 28px 34px 48px; }
    section { margin-bottom: 36px; }
    h2 { margin: 0 0 10px 0; font-size: 28px; }
    h3 { margin: 22px 0 8px 0; font-size: 20px; }
    p { line-height: 1.6; }
    .section-lead { margin: 8px 0 0; color: #334155; max-width: 980px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 16px 0 18px; }
    .card { background: #fff; border-radius: 14px; padding: 16px; box-shadow: 0 6px 20px rgba(15,23,42,0.06); border: 1px solid #e2e8f0; }
    .card .label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #475569; }
    .card .value { font-size: 24px; font-weight: 700; margin-top: 6px; }
    .panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px; box-shadow: 0 8px 26px rgba(15,23,42,0.05); margin-top: 14px; }
    .explain { margin: 10px 0 14px; padding: 12px 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; }
    .explain-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px 14px; }
    .explain-item { font-size: 12.5px; line-height: 1.45; color: #334155; }
    .explain-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #0f4c5c; margin-bottom: 4px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .report-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .report-table th, .report-table td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; }
    .report-table th { background: #f8fafc; }
    .heatmap th, .heatmap td { text-align: center; }
    .chart { width: 100%; height: auto; display: block; }
    ul.flat { padding-left: 18px; }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } nav { position: static; height: auto; } .two-col { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="layout">
  <nav>
    <h1>Embedding Report</h1>
    <p>Diagnostico integral de cobertura, calidad y utilidad sobre los embeddings finales, separando validacion honesta de diagnosticos post-export.</p>
    <a href="#resumen">Resumen</a><a href="#cobertura">Cobertura y Salud</a><a href="#utilidad">Utilidad</a><a href="#negocio">Negocios</a><a href="#usuarios">Usuarios</a><a href="#amistades">Amistades</a><a href="#recomendaciones">Recomendaciones</a>
  </nav>
  <main>
    <section id="resumen">
      <h2>Resumen Ejecutivo</h2>
      <p>El encoder profundo conserva la mejor validacion honesta disponible en el entrenamiento, pero los scorers sobre embeddings exportados solo se deben leer como diagnosticos post-export. En negocio, el espacio profundo sigue mostrando coherencia local util, y en social hay homofilia positiva tras controlar por actividad.</p>
      <div class="cards">{% for card in coverage.kpis %}<div class="card"><div class="label">{{ card.label }}</div><div class="value">{{ card.value }}</div></div>{% endfor %}<div class="card"><div class="label">MAE temporal oficial deep</div><div class="value">{{ "%.4f"|format(coverage.deep_summary.best_val_mae) }}</div></div><div class="card"><div class="label">MAE post-export deep</div><div class="value">{{ "%.4f"|format(utility.diagnostic_summary.deep_mae) }}</div></div><div class="card"><div class="label">Usuarios history-based</div><div class="value">{{ "{:,}".format(coverage.deep_summary.embedding_source_counts.history) }}</div></div><div class="card"><div class="label">Uplift social deep</div><div class="value">{{ "%.4f"|format(friends.summary_table.iloc[0].uplift if not friends.summary_table.empty else 0.0) }}</div></div></div>
    </section>
    <section id="cobertura"><h2>Cobertura y Salud Del Espacio</h2><div class="two-col"><div class="panel"><h3>Fuente del embedding por usuario</h3>{{ coverage.source_counts_html | safe }}</div><div class="panel"><h3>Bandas de historial</h3>{{ coverage.band_counts_html | safe }}</div></div><div class="panel"><h3>Composicion `history_band x embedding_source`</h3>{{ coverage.source_band_chart | safe }}</div><div class="two-col"><div class="panel"><h3>Bloques del embedding completo de negocio</h3>{{ coverage.business_blocks_html | safe }}</div><div class="panel"><h3>Salud de las representaciones</h3>{{ coverage.health_html | safe }}</div></div></section>
    <section id="utilidad"><h2>Utilidad y Diagnostico</h2>{{ utility.section_intro_html | safe }}<div class="two-col"><div class="panel"><h3>Validacion honesta del encoder profundo</h3>{{ utility.honest_html | safe }}</div><div class="panel"><h3>Diagnostico post-export sobre el snapshot temporal</h3>{{ utility.diagnostic_html | safe }}</div></div><div class="two-col"><div class="panel"><h3>AUC de preferencia</h3>{{ utility.preference_html | safe }}</div><div class="panel"><h3>MAE por banda de historial</h3>{{ utility.band_chart | safe }}</div></div></section>
    <section id="negocio"><h2>Embeddings de Negocio</h2><div class="panel"><h3>Coherencia local global</h3>{{ business.overall_html | safe }}</div><div class="two-col"><div class="panel"><h3>Coherencia por categoria</h3>{{ business.category_html | safe }}</div><div class="panel"><h3>Mapa PCA de negocios profundos</h3>{{ business.scatter_svg | safe }}</div></div><div class="panel"><h3>Similitud entre centroides de categorias</h3>{{ business.heatmap_html | safe }}</div></section>
    <section id="usuarios"><h2>Embeddings de Usuario</h2><div class="two-col"><div class="panel"><h3>Consistencia manual vs deep</h3><p><strong>Correlacion de distancias:</strong> {{ "%.4f"|format(user_consistency.pairwise_corr) }}</p><p><strong>Solape medio de vecinos top-10:</strong> {{ "%.4f"|format(user_consistency.neighbor_overlap_mean) }}</p><p><strong>Muestra analizada:</strong> {{ "{:,}".format(user_consistency.n_users) }} usuarios con al menos 2 reviews.</p></div><div class="panel"><h3>Clustering historico</h3><p><strong>Usuarios usados:</strong> {{ "{:,}".format(clusters.historical.n_users) }}</p><p><strong>Silhouette:</strong> {{ "%.4f"|format(clusters.historical.silhouette) if clusters.historical.silhouette == clusters.historical.silhouette else "n/a" }}</p>{{ clusters.historical.table_html | safe }}</div></div><div class="panel"><h3>Mapa PCA de clusters historicos</h3>{{ clusters.historical.scatter_svg | safe }}</div><div class="two-col"><div class="panel"><h3>Clustering cold-start</h3><p><strong>Usuarios usados:</strong> {{ "{:,}".format(clusters.cold_start.n_users) }}</p><p><strong>Silhouette:</strong> {{ "%.4f"|format(clusters.cold_start.silhouette) if clusters.cold_start.silhouette == clusters.cold_start.silhouette else "n/a" }}</p>{{ clusters.cold_start.table_html | safe }}</div><div class="panel"><h3>Mapa PCA cold-start</h3>{{ clusters.cold_start.scatter_svg | safe }}</div></div></section>
    <section id="amistades"><h2>Amistades y Homofilia Social</h2><div class="cards"><div class="card"><div class="label">Usuarios con amigos validos</div><div class="value">{{ "{:,}".format(friends.graph_summary.n_users_with_valid_friend_edges) }}</div></div><div class="card"><div class="label">Aristas dirigidas validas</div><div class="value">{{ "{:,}".format(friends.graph_summary.directed_valid_edges) }}</div></div><div class="card"><div class="label">p90 grado social</div><div class="value">{{ "%.0f"|format(friends.graph_summary.degree_p90) }}</div></div><div class="card"><div class="label">maximo grado</div><div class="value">{{ "{:,}".format(friends.graph_summary.degree_max) }}</div></div></div><div class="two-col"><div class="panel"><h3>Resumen friend vs matched non-friend</h3>{{ friends.summary_html | safe }}</div><div class="panel"><h3>Subconjunto history-history</h3>{{ friends.primary_html | safe }}</div></div><div class="panel"><h3>Visualizacion del uplift social</h3>{{ friends.chart_svg | safe }}</div></section>
    <section id="recomendaciones"><h2>Recomendaciones Para Competicion</h2><div class="panel"><ul class="flat"><li>Usar `user_deep + business_deep` como espacio principal para scoring y recuperacion semantica, sobre todo en usuarios con `history_band >= 2`.</li><li>Mantener `user_manual_profile + business_full` como baseline robusto y como fuente complementaria de explicabilidad.</li><li>Separar explicitamente `cold-start`, `single-review` e `historical` en el pipeline final.</li><li>Introducir la similitud social como senal auxiliar solo en `history-history` o como regularizacion.</li><li>El mayor retorno parece estar en mejorar cold-start de usuario y en reforzar el scorer final.</li></ul></div></section>
  </main>
</div>
</body>
</html>
        """
    )
    rendered = template.render(coverage=coverage, utility=utility, business=business_metrics, user_consistency=user_consistency, clusters=clusters, friends=friends)
    notes = [
        ("<h2>Resumen Ejecutivo</h2>", coverage["summary_intro_html"]),
        ("<h2>Cobertura y Salud Del Espacio</h2>", coverage["section_intro_html"]),
        ("<h3>Fuente del embedding por usuario</h3>", coverage["source_counts_note_html"]),
        ("<h3>Bandas de historial</h3>", coverage["band_counts_note_html"]),
        ("<h3>Composicion `history_band x embedding_source`</h3>", coverage["source_band_note_html"]),
        ("<h3>Bloques del embedding completo de negocio</h3>", coverage["business_blocks_note_html"]),
        ("<h3>Salud de las representaciones</h3>", coverage["health_note_html"]),
        ("<h2>Utilidad y Diagnostico</h2>", utility["section_intro_html"]),
        ("<h3>Validacion honesta del encoder profundo</h3>", utility["honest_note_html"]),
        ("<h3>Diagnostico post-export sobre el snapshot temporal</h3>", utility["diagnostic_note_html"]),
        ("<h3>AUC de preferencia</h3>", utility["preference_note_html"]),
        ("<h3>MAE por banda de historial</h3>", utility["band_note_html"]),
        ("<h2>Embeddings de Negocio</h2>", business_metrics["section_intro_html"]),
        ("<h3>Coherencia local global</h3>", business_metrics["overall_note_html"]),
        ("<h3>Coherencia por categoria</h3>", business_metrics["category_note_html"]),
        ("<h3>Mapa PCA de negocios profundos</h3>", business_metrics["scatter_note_html"]),
        ("<h3>Similitud entre centroides de categorias</h3>", business_metrics["heatmap_note_html"]),
        ("<h2>Embeddings de Usuario</h2>", user_consistency["section_intro_html"]),
        ("<h3>Consistencia manual vs deep</h3>", user_consistency["note_html"]),
        ("<h3>Clustering historico</h3>", clusters["historical"]["note_html"]),
        ("<h3>Mapa PCA de clusters historicos</h3>", clusters["historical"]["scatter_note_html"]),
        ("<h3>Clustering cold-start</h3>", clusters["cold_start"]["note_html"]),
        ("<h3>Mapa PCA cold-start</h3>", clusters["cold_start"]["scatter_note_html"]),
        ("<h2>Amistades y Homofilia Social</h2>", friends["section_intro_html"]),
        ("<h3>Resumen friend vs matched non-friend</h3>", friends["summary_note_html"]),
        ("<h3>Subconjunto history-history</h3>", friends["primary_note_html"]),
        ("<h3>Visualizacion del uplift social</h3>", friends["chart_note_html"]),
        ("<h2>Recomendaciones Para Competicion</h2>", recommendations["note_html"]),
    ]
    return inject_html_snippets(rendered, notes)


def save_outputs_v2(report_dir: Path, *, coverage: dict[str, Any], utility: dict[str, Any], business_metrics: dict[str, Any], clusters: dict[str, Any], friends: dict[str, Any], html_report: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "embedding_quality_report.html").write_text(html_report, encoding="utf-8")
    utility["honest_table"].to_csv(report_dir / "utility_honest_validation.csv", index=False)
    utility["diagnostic_table"].to_csv(report_dir / "utility_post_export_diagnostics.csv", index=False)
    utility["diagnostic_table"].to_csv(report_dir / "utility_summary.csv", index=False)
    utility["preference_table"].to_csv(report_dir / "utility_preference_auc.csv", index=False)
    utility["band_table"].to_csv(report_dir / "utility_band_mae.csv", index=False)
    business_metrics["overall_summary"].to_csv(report_dir / "business_coherence_overall.csv", index=False)
    business_metrics["category_summary"].to_csv(report_dir / "business_coherence_by_category.csv", index=False)
    business_metrics["centroid_similarity"].to_csv(report_dir / "business_centroid_similarity.csv")
    clusters["historical"]["cluster_profiles"].to_csv(report_dir / "historical_cluster_profiles.csv", index=False)
    clusters["cold_start"]["cluster_profiles"].to_csv(report_dir / "cold_start_cluster_profiles.csv", index=False)
    friends["summary_table"].to_csv(report_dir / "friend_similarity_summary.csv", index=False)
    friends["primary_table"].to_csv(report_dir / "friend_similarity_primary.csv", index=False)
    friends["pair_frame"].to_csv(report_dir / "friend_similarity_pairs.csv", index=False)
    summary = {
        "honest_deep_mae": float(coverage["deep_summary"]["best_val_mae"]),
        "honest_deep_rmse": float(coverage["deep_summary"]["best_val_rmse"]),
        "deep_best_val_mae": float(coverage["deep_summary"]["best_val_mae"]),
        "deep_best_val_rmse": float(coverage["deep_summary"]["best_val_rmse"]),
        "post_export_manual_mae": float(utility["diagnostic_summary"]["manual_mae"]),
        "post_export_manual_rmse": float(utility["diagnostic_summary"]["manual_rmse"]),
        "post_export_deep_mae": float(utility["diagnostic_summary"]["deep_mae"]),
        "post_export_deep_rmse": float(utility["diagnostic_summary"]["deep_rmse"]),
        "social_uplift_deep": float(friends["summary_table"].iloc[0]["uplift"]) if not friends["summary_table"].empty else None,
        "historical_cluster_silhouette": float(clusters["historical"]["silhouette"]) if clusters["historical"]["silhouette"] == clusters["historical"]["silhouette"] else None,
        "cold_cluster_silhouette": float(clusters["cold_start"]["silhouette"]) if clusters["cold_start"]["silhouette"] == clusters["cold_start"]["silhouette"] else None,
    }
    (report_dir / "embedding_quality_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifacts = load_artifacts(args.artifacts_root)
    users_df = load_users(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    users_schema_problems: list[str] = []
    validate_required_columns(users_df, "users_table", REQUIRED_COLUMNS["users_table"], users_schema_problems)
    raise_artifact_issues(args.data_dir, users_schema_problems)

    coverage = coverage_section(artifacts, users_df)
    utility = build_utility_metrics(artifacts, train_reviews)
    business_metrics = build_business_semantic_metrics(artifacts, args.top_k_neighbors, args.sample_business_pca, args.business_anchor_per_category)
    user_consistency = compute_user_similarity_consistency(artifacts, args.sample_user_consistency)
    dominant_category_df = build_macro_history_lookup(train_reviews, add_business_semantics(artifacts.business_table))
    clusters = build_cluster_profiles(artifacts.user_deep_table, artifacts.user_deep, dominant_category_df, args.sample_user_history_clustering, args.sample_cold_clustering)
    friends = build_friend_similarity_analysis(artifacts, users_df, args.sample_friend_anchors, args.friend_pairs_per_anchor)

    coverage["source_counts_html"] = styled_table(coverage["source_counts"])
    coverage["band_counts_html"] = styled_table(coverage["band_counts"])
    pivot = coverage["source_band"].pivot(index="history_band", columns="embedding_source", values="count").fillna(0.0).reset_index()
    coverage["source_band_chart"] = svg_grouped_bar(pivot, "history_band", [col for col in pivot.columns if col != "history_band"])
    coverage["business_blocks_html"] = styled_table(coverage["business_blocks"], float_cols=["coverage"])
    health_rows = []
    for name, metrics in coverage["health"].items():
        row = {"space": name}
        row.update(metrics)
        health_rows.append(row)
    health_df = pd.DataFrame(health_rows)
    coverage["health_html"] = styled_table(health_df, float_cols=[col for col in health_df.columns if col != "space"])

    utility["honest_html"] = styled_table(utility["honest_table"], float_cols=["mae", "rmse"])
    utility["diagnostic_html"] = styled_table(utility["diagnostic_table"], float_cols=["mae", "rmse"])
    utility["preference_html"] = styled_table(utility["preference_table"], float_cols=["pairwise_auc"])
    utility["band_chart"] = svg_grouped_bar(utility["band_table"], "history_band", ["manual_mae", "deep_mae"])

    business_metrics["overall_html"] = styled_table(business_metrics["overall_summary"], float_cols=["same_category_ratio", "same_city_ratio", "category_entropy"])
    business_metrics["category_html"] = styled_table(business_metrics["category_summary"], float_cols=["same_category_ratio", "same_city_ratio", "category_entropy"])
    business_metrics["scatter_svg"] = svg_scatter(business_metrics["scatter"], "x", "y", "category")
    business_metrics["heatmap_html"] = heatmap_table(business_metrics["centroid_similarity"])

    clusters["historical"]["table_html"] = styled_table(clusters["historical"]["cluster_profiles"], float_cols=["history_median", "tenure_mean", "elite_rate", "fans_mean_z", "social_capital_mean_z"])
    clusters["historical"]["scatter_svg"] = svg_scatter(clusters["historical"]["scatter"], "x", "y", "cluster")
    clusters["cold_start"]["table_html"] = styled_table(clusters["cold_start"]["cluster_profiles"], float_cols=["history_median", "tenure_mean", "elite_rate", "fans_mean_z", "social_capital_mean_z"])
    clusters["cold_start"]["scatter_svg"] = svg_scatter(clusters["cold_start"]["scatter"], "x", "y", "cluster")

    friends["summary_html"] = styled_table(friends["summary_table"], float_cols=["friend_mean_cos", "matched_non_friend_mean_cos", "uplift", "paired_win_rate"])
    friends["primary_html"] = styled_table(friends["primary_table"], float_cols=["uplift", "paired_win_rate"])
    friends["chart_svg"] = svg_grouped_bar(friends["summary_table"][["space", "friend_mean_cos", "matched_non_friend_mean_cos", "uplift"]].copy(), "space", ["friend_mean_cos", "matched_non_friend_mean_cos", "uplift"])

    coverage["summary_intro_html"] = section_intro_html(
        "Empieza por aqui: este bloque resume si los embeddings cubren bien el snapshot y si la calidad cambia mucho entre history-based, metadata-only y fallback."
    )
    coverage["section_intro_html"] = section_intro_html(
        "Aqui se ve cuanta poblacion real entra en cada familia de embedding y si la representacion cae en fallbacks o en usuarios con poco historial."
    )
    coverage["source_counts_note_html"] = explainer_html(
        measures="Cuenta cuantos usuarios caen en history, metadata_only o default_only.",
        read="Un reparto sano no depende solo de fallback; deberia haber masa suficiente en history si el encoder aprendio senal real.",
        good_bad="Bueno: history domina o al menos tiene peso relevante. Malo: metadata_only domina casi todo el universo.",
        warning="El total puede verse bien aunque la calidad por fuente sea muy desigual; mira tambien la banda de historial.",
    )
    coverage["band_counts_note_html"] = explainer_html(
        measures="Distribucion de usuarios por cantidad de reviews historicas.",
        read="Sirve para ver si el problema esta concentrado en cold-start o si hay una base amplia de usuarios con historial.",
        good_bad="Bueno: hay masa en bandas 2-5, 6-20 y >20. Malo: todo queda en 0-1 reviews.",
        warning="La mediana puede ser baja aunque existan usuarios muy activos; no dejes que los outliers escondan el cold-start.",
    )
    coverage["source_band_note_html"] = explainer_html(
        measures="Cruza fuente del embedding con banda de historial.",
        read="Te dice si los usuarios con mas historial acaban en embeddings history-based o si se apoyan en metadata.",
        good_bad="Bueno: history se concentra en bandas con historial y metadata_only en cold-start. Malo: history aparece casi solo en 0-1 reviews.",
        warning="No compares columnas sin mirar el tamano de cada celda; una banda pequena puede parecer ruidosa por pura muestra.",
    )
    coverage["business_blocks_note_html"] = explainer_html(
        measures="Resume el tamano y la cobertura de los bloques del embedding completo de negocio.",
        read="Cada fila refleja un bloque semantico del vector manual; te ayuda a ver si la representacion esta equilibrada o cargada en un bloque.",
        good_bad="Bueno: bloques completos y con cobertura similar. Malo: muchos bloques vacios o con densidad extrema desigual.",
        warning="La cobertura no implica utilidad; un bloque puede ser denso y aun asi poco informativo.",
    )
    coverage["health_note_html"] = explainer_html(
        measures="Diagnostica norm, varianza y dimensiones muertas en cada espacio de embedding.",
        read="Normas muy bajas, muchas dimensiones muertas o varianza casi nula suelen indicar colapso o representacion poco expresiva.",
        good_bad="Bueno: pocas dimensiones muertas y normas estables. Malo: demasiados ceros, varianza casi nula o dispersion extrema.",
        warning="Una norma alta no es automaticamente buena; puede ser solo una escala mal calibrada.",
    )

    utility["section_intro_html"] = section_intro_html(
        "Este bloque separa la validacion honesta del encoder profundo de los diagnosticos post-export sobre embeddings fijos."
    )
    utility["honest_note_html"] = explainer_html(
        measures="Resume la validacion temporal interna del encoder profundo.",
        read="Este numero es la referencia honesta: procede del entrenamiento del encoder y no depende de un scorer lineal post-export.",
        good_bad="Bueno: la MAE y RMSE del checkpoint original son competitivos. Malo: empeoran frente a versiones mas simples.",
        warning="No mezcles este valor con los scorers lineales sobre embeddings exportados; miden cosas distintas.",
    )
    utility["diagnostic_note_html"] = explainer_html(
        measures="Resume scorers lineales sobre embeddings exportados y evaluados en un split temporal posterior.",
        read="Este bloque sirve para comparar espacios exportados bajo el mismo protocolo, pero no es una validacion honesta del encoder.",
        good_bad="Bueno: baja MAE/RMSE en manual o deep. Malo: no mejora frente al baseline o queda inestable por banda.",
        warning="La etiqueta es diagnostico post-export: no usar como reemplazo del score oficial de entrenamiento.",
    )
    utility["preference_note_html"] = explainer_html(
        measures="Mide cuantas veces los items bien valorados quedan por encima de los mal valorados dentro de cada usuario.",
        read="Es una senal de ranking: cuanto mas alto, mejor separa el espacio las preferencias relativas.",
        good_bad="Bueno: valores claramente por encima de 0.5. Malo: cerca de azar o por debajo de 0.5.",
        warning="No sustituye MAE; solo complementa la lectura de ordenacion interna del espacio.",
    )
    utility["band_note_html"] = explainer_html(
        measures="Desglosa el MAE por cantidad de historial del usuario.",
        read="Sirve para comprobar si el modelo gana sobre todo en usuarios con historial o si tambien mejora cold-start.",
        good_bad="Bueno: la curva baja al aumentar historial y no se dispara en 0-1 reviews. Malo: mucha mejora solo en usuarios faciles.",
        warning="Las bandas con pocas muestras pueden oscilar bastante; interpreta con cautela los extremos.",
    )

    business_metrics["section_intro_html"] = section_intro_html(
        "Aqui miramos si el espacio de negocio conserva categoria, ciudad y vecindades semanticas utiles."
    )
    business_metrics["overall_note_html"] = explainer_html(
        measures="Resume la coherencia local media de vecinos para business_deep y business_full.",
        read="Si el ratio de misma categoria y misma ciudad es alto, el espacio agrupa negocios parecidos de forma util.",
        good_bad="Bueno: high same_category_ratio y mismo_city_ratio razonable. Malo: vecindarios casi aleatorios.",
        warning="Un exceso de similitud tambien puede significar espacio demasiado colapsado; no persigas solo ratios altos.",
    )
    business_metrics["category_note_html"] = explainer_html(
        measures="Desglosa la coherencia local por categoria ancla.",
        read="Te dice en que categorias el embedding funciona mejor o peor y donde la semantica es mas estable.",
        good_bad="Bueno: categorias grandes y claras con buen ratio. Malo: categorias con mezcla excesiva o muy poco soporte.",
        warning="Categorias minoritarias pueden parecer peores solo por tener menos ejemplos.",
    )
    business_metrics["scatter_note_html"] = explainer_html(
        measures="Proyecta negocios a 2D con PCA para ver si aparecen nubes separadas por categoria.",
        read="No busques fronteras perfectas; busca estructura, solapamiento razonable y categorias reconocibles.",
        good_bad="Bueno: agrupaciones visibles y ejes con sentido. Malo: nube totalmente amorfa o un unico bloque.",
        warning="PCA comprime mucho; no interpretes distancias visuales como si fueran el espacio completo.",
    )
    business_metrics["heatmap_note_html"] = explainer_html(
        measures="Mide similitud entre centroides de categorias en el espacio deep.",
        read="Valores altos entre categorias afines son esperables; valores altos entre categorias lejanas pueden indicar colapso.",
        good_bad="Bueno: categorias cercanas con similitud alta y categorias distintas mas separadas. Malo: todo se parece a todo.",
        warning="La similitud de centroides no sustituye a la coherencia por vecino; es solo una vista agregada.",
    )

    user_consistency["section_intro_html"] = section_intro_html(
        "Esta parte compara si el espacio profundo de usuarios conserva una geometria parecida al manual o si aprende una estructura nueva."
    )
    user_consistency["note_html"] = explainer_html(
        measures="Compara distancias y vecinos entre user_manual_profile y user_deep.",
        read="Si la correlacion y el solape de vecinos son altos, ambos espacios cuentan una historia parecida; si bajan, el deep esta capturando otra senal.",
        good_bad="Bueno: correlacion y overlap razonables pero no identicos. Malo: cero acuerdo o una copia trivial.",
        warning="Esta comparacion solo usa usuarios con historial minimo; no representa por igual a cold-start.",
    )
    clusters["historical"]["note_html"] = explainer_html(
        measures="Resume clusters sobre usuarios con historial real.",
        read="Te ayuda a ver segmentos interpretables y si el embedding separa grupos con perfiles distintos.",
        good_bad="Bueno: clusters de tamano razonable con silhouette positiva. Malo: clusters diminutos o muy mezclados.",
        warning="Silhouette alta no garantiza utilidad de negocio; prioriza interpretabilidad y estabilidad.",
    )
    clusters["historical"]["scatter_note_html"] = explainer_html(
        measures="Muestra la proyeccion 2D de usuarios historicos por cluster.",
        read="Sirve para ver si los clusters se separan visualmente o si solo existen en alta dimension.",
        good_bad="Bueno: nubes relativamente compactas. Malo: cluster IDs totalmente entremezclados.",
        warning="La proyeccion 2D puede ocultar estructura real o inventar separaciones aparentes.",
    )
    clusters["cold_start"]["note_html"] = explainer_html(
        measures="Resume clusters de usuarios resueltos por metadata_only o default_only.",
        read="Este bloque permite ver si el fallback produce segmentos interpretable sin depender de historial.",
        good_bad="Bueno: segmentos ligados a tenure, fans o elite con lectura clara. Malo: un grupo caotico sin perfil.",
        warning="No compares directamente su silhouette con la de historicos; son poblaciones distintas.",
    )
    clusters["cold_start"]["scatter_note_html"] = explainer_html(
        measures="Muestra la proyeccion 2D del cold-start.",
        read="Debe usarse para buscar patrones de metadata, no para exigir separacion perfecta.",
        good_bad="Bueno: grupos compactos y consistentes con metadata. Malo: nube sin forma o clusters arbitrarios.",
        warning="El cold-start suele estar condicionado por pocas variables; la lectura debe ser prudente.",
    )

    friends["section_intro_html"] = section_intro_html(
        "Aqui comprobamos si la homofilia social existe de verdad o si desaparece cuando controlamos por actividad y cobertura."
    )
    friends["summary_note_html"] = explainer_html(
        measures="Compara similitud entre pares de amigos y no-amigos emparejados.",
        read="Si el uplift es positivo, los amigos tienden a caer mas cerca en el espacio. Si no, la red social aporta poca senal geometrica.",
        good_bad="Bueno: uplift positivo y win rate por encima de 0.5. Malo: uplift nulo o negativo.",
        warning="Siempre lee esto junto con el control por actividad; la homofilia aparente puede venir de hubs muy activos.",
    )
    friends["primary_note_html"] = explainer_html(
        measures="Repite el analisis solo en pares history-history.",
        read="Es el subconjunto mas limpio para ver si la similitud social se mantiene cuando ambos usuarios tienen historial real.",
        good_bad="Bueno: uplift parecido o mejor que el global. Malo: la senal desaparece en el subconjunto limpio.",
        warning="Es una muestra mas pequena, asi que la variabilidad sube; evita conclusiones demasiado fuertes por un unico numero.",
    )
    friends["chart_note_html"] = explainer_html(
        measures="Visualiza la diferencia de similitud entre amigos, no-amigos y uplift.",
        read="Te deja ver rapido si la separacion es robusta o si depende de unos pocos casos extremos.",
        good_bad="Bueno: amigos por encima de no-amigos de forma consistente. Malo: barras solapadas o muy inestables.",
        warning="Un grafico agregado puede esconder sesgo de hubs; complementalo con la tabla.",
    )

    recommendations = {
        "note_html": section_intro_html(
            "Cierra el informe con lectura operativa: que espacio usar, donde hay riesgo y donde conviene invertir esfuerzo para la competicion."
        )
    }

    html_report = build_report_html_v2(coverage=coverage, utility=utility, business_metrics=business_metrics, user_consistency=user_consistency, clusters=clusters, friends=friends, recommendations=recommendations)
    save_outputs_v2(args.report_dir, coverage=coverage, utility=utility, business_metrics=business_metrics, clusters=clusters, friends=friends, html_report=html_report)
    print(json.dumps({"report_dir": str(args.report_dir), "html_report": str(args.report_dir / "embedding_quality_report.html")}, indent=2))


if __name__ == "__main__":
    main()
