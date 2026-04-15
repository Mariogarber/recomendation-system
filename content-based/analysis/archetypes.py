from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ARCHETYPE_METRIC_COLUMNS = [
    "user_average_stars_mean",
    "user_review_count_log1p_mean",
    "user_tenure_years_mean",
    "user_total_votes_log1p_mean",
    "user_engagement_log1p_mean",
    "user_friends_log1p_mean",
    "user_elite_years_count_mean",
    "user_compliment_log1p_total_mean",
    "user_compliment_nonzero_count_mean",
]

DISPLAY_LABELS = {
    "user_average_stars_mean": "Average stars",
    "user_review_count_log1p_mean": "Review count log1p",
    "user_tenure_years_mean": "Tenure years",
    "user_total_votes_log1p_mean": "Total votes log1p",
    "user_engagement_log1p_mean": "Engagement log1p",
    "user_friends_log1p_mean": "Friends log1p",
    "user_elite_years_count_mean": "Elite years",
    "user_compliment_log1p_total_mean": "Compliments log1p",
    "user_compliment_nonzero_count_mean": "Compliment nonzero count",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _label_from_zscore(value: float, high: str, low: str, neutral: str) -> str:
    if value >= 1.0:
        return high
    if value <= -1.0:
        return low
    return neutral


def _build_archetype_descriptors(archetypes: pd.DataFrame, zscores: pd.DataFrame) -> pd.DataFrame:
    descriptors = []
    for idx, row in archetypes.iterrows():
        z = zscores.loc[idx]
        descriptors.append(
            {
                "user_archetype_id": row["user_archetype_id"],
                "summary_label": " | ".join(
                    [
                        _label_from_zscore(z["user_average_stars_mean"], "muy positiva", "muy critica", "rating medio"),
                        _label_from_zscore(z["user_review_count_log1p_mean"], "muy activa", "poco activa", "actividad media"),
                        _label_from_zscore(z["user_tenure_years_mean"], "veterana", "reciente", "antiguedad media"),
                        _label_from_zscore(z["user_engagement_log1p_mean"], "muy implicada", "poco implicada", "implicacion media"),
                    ]
                ),
                "reputation_descriptor": _label_from_zscore(
                    z["user_average_stars_mean"], "optimista", "exigente", "equilibrada"
                ),
                "activity_descriptor": _label_from_zscore(
                    z["user_review_count_log1p_mean"], "hiperactiva", "ocasional", "regular"
                ),
                "tenure_descriptor": _label_from_zscore(
                    z["user_tenure_years_mean"], "veterana", "nueva", "estable"
                ),
                "social_descriptor": _label_from_zscore(
                    z["user_friends_log1p_mean"], "muy social", "poco social", "social media"
                ),
                "prestige_descriptor": _label_from_zscore(
                    z["user_elite_years_count_mean"] + z["user_compliment_log1p_total_mean"],
                    "alta senal elite",
                    "baja senal elite",
                    "senal elite media",
                ),
            }
        )
    return pd.DataFrame(descriptors)


def _build_projection(archetypes: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    x = scaler.fit_transform(archetypes[ARCHETYPE_METRIC_COLUMNS])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)
    projection = archetypes[["user_archetype_id", "n_users"]].copy()
    projection["pc1"] = coords[:, 0]
    projection["pc2"] = coords[:, 1]
    projection["explained_variance_ratio_pc1"] = float(pca.explained_variance_ratio_[0])
    projection["explained_variance_ratio_pc2"] = float(pca.explained_variance_ratio_[1])
    return projection


def _safe_round(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].astype(float).round(decimals)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive HTML dashboard for raw-router user archetypes.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "lgbm_raw_router_v1",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root
    save_path = args.save_path or artifact_root / "archetype_dashboard.html"

    archetypes = pd.read_csv(artifact_root / "archetype_profiles.csv")
    known_importance = pd.read_csv(artifact_root / "known_feature_importance.csv")
    cold_importance = pd.read_csv(artifact_root / "cold_feature_importance.csv")
    validation_summary = _load_json(artifact_root / "validation_summary.json")
    feature_manifest = _load_json(artifact_root / "feature_manifest.json")
    discarded_variables = _load_json(artifact_root / "discarded_variables.json")
    submission_summary = _load_json(artifact_root / "submission_summary.json")

    zscores = pd.DataFrame(
        StandardScaler().fit_transform(archetypes[ARCHETYPE_METRIC_COLUMNS]),
        columns=ARCHETYPE_METRIC_COLUMNS,
        index=archetypes.index,
    )
    descriptors = _build_archetype_descriptors(archetypes, zscores)
    projection = _build_projection(archetypes)

    archetypes_enriched = archetypes.merge(descriptors, on="user_archetype_id", how="left").merge(
        projection[["user_archetype_id", "pc1", "pc2"]],
        on="user_archetype_id",
        how="left",
    )
    archetypes_enriched["size_pct"] = archetypes_enriched["n_users"] / archetypes_enriched["n_users"].sum()

    numeric_table = _safe_round(
        archetypes_enriched,
        [
            "metadata_completeness_mean",
            "metadata_sparse_rate",
            "user_average_stars_mean",
            "user_review_count_log1p_mean",
            "user_tenure_years_mean",
            "user_total_votes_log1p_mean",
            "user_engagement_log1p_mean",
            "user_friends_log1p_mean",
            "user_elite_years_count_mean",
            "user_compliment_log1p_total_mean",
            "user_compliment_nonzero_count_mean",
            "pc1",
            "pc2",
            "size_pct",
        ],
    )

    payload = {
        "summary": {
            "n_archetypes": int(len(archetypes)),
            "n_users_total": int(archetypes["n_users"].sum()),
            "router_validation_mae_rounded": float(validation_summary["router_validation_mae_rounded"]),
            "router_band0_mae": float(validation_summary["band_metrics_router"][0]["mae"]),
            "known_branch_rows": int(submission_summary["known_branch_rows"]),
            "cold_branch_rows": int(submission_summary["cold_branch_rows"]),
            "pc1_var": float(projection["explained_variance_ratio_pc1"].iloc[0]),
            "pc2_var": float(projection["explained_variance_ratio_pc2"].iloc[0]),
        },
        "archetypes": numeric_table.to_dict(orient="records"),
        "zscore_table": _safe_round(
            pd.concat([archetypes[["user_archetype_id"]], zscores], axis=1),
            ARCHETYPE_METRIC_COLUMNS,
        ).to_dict(orient="records"),
        "metric_columns": ARCHETYPE_METRIC_COLUMNS,
        "display_labels": DISPLAY_LABELS,
        "known_top_features": known_importance.head(20).round(3).to_dict(orient="records"),
        "cold_top_features": cold_importance.head(25).round(3).to_dict(orient="records"),
        "feature_manifest": feature_manifest,
        "discarded_variables": discarded_variables,
    }

    plotly_js = get_plotlyjs()
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dashboard de Arquetipos de Usuario</title>
  <style>
    :root {{
      --bg: #f5f2ea;
      --panel: #fffdf8;
      --ink: #1f2b21;
      --muted: #65756a;
      --accent: #0b6e4f;
      --accent-soft: #d8efe7;
      --line: #d8d2c4;
      --warm: #d97706;
      --cool: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background: linear-gradient(180deg, #f7f3ea 0%, #f0ece2 100%);
      color: var(--ink);
    }}
    .shell {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 10px 28px rgba(31, 43, 33, 0.06);
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 32px; line-height: 1.1; }}
    h2 {{ font-size: 18px; }}
    .sub {{
      color: var(--muted);
      line-height: 1.5;
      margin-bottom: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .stat {{
      background: #faf7ef;
      border: 1px solid #e8e1d2;
      border-radius: 14px;
      padding: 12px;
    }}
    .stat .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat .value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
    }}
    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .controls label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    select, input {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      color: var(--ink);
      margin-bottom: 12px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .chip {{
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .detail-card {{
      background: #faf7ef;
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid #ece4d6;
    }}
    .detail-card .key {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .detail-card .val {{
      font-size: 18px;
      font-weight: 700;
    }}
    .main {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart {{
      min-height: 420px;
    }}
    .wide-chart {{
      min-height: 500px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #eee5d6;
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #fbf8f0;
      z-index: 1;
    }}
    .table-wrap {{
      max-height: 340px;
      overflow: auto;
      border: 1px solid #ebe4d6;
      border-radius: 14px;
    }}
    .table-row-active {{
      background: #edf8f4;
    }}
    .mono {{
      font-family: Consolas, monospace;
    }}
    .footnote {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-top: 10px;
    }}
    @media (max-width: 1200px) {{
      .hero, .layout, .chart-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel">
        <h1>Dashboard de Arquetipos de Usuario</h1>
        <div class="sub">
          Explorador interactivo de los <strong>{payload["summary"]["n_archetypes"]}</strong> arquetipos aprendidos para el router de cold start.
          El objetivo es comprobar si los grupos tienen sentido de negocio y qué variables están moviendo la rama `cold_model`.
        </div>
        <div class="stats">
          <div class="stat"><div class="label">Usuarios cubiertos</div><div class="value">{payload["summary"]["n_users_total"]:,}</div></div>
          <div class="stat"><div class="label">MAE Router Local</div><div class="value">{payload["summary"]["router_validation_mae_rounded"]:.4f}</div></div>
          <div class="stat"><div class="label">MAE Banda 0</div><div class="value">{payload["summary"]["router_band0_mae"]:.4f}</div></div>
          <div class="stat"><div class="label">Filas rama known</div><div class="value">{payload["summary"]["known_branch_rows"]:,}</div></div>
          <div class="stat"><div class="label">Filas rama cold</div><div class="value">{payload["summary"]["cold_branch_rows"]:,}</div></div>
          <div class="stat"><div class="label">PCA var explicada</div><div class="value">{payload["summary"]["pc1_var"] + payload["summary"]["pc2_var"]:.1%}</div></div>
        </div>
      </div>
      <div class="panel">
        <h2>Qué mirar aquí</h2>
        <div class="sub">
          1. Si los arquetipos grandes son coherentes. 2. Si los extremos de rating y actividad están bien separados.
          3. Si las variables más importantes de la rama cold encajan con los grupos aprendidos.
        </div>
        <div class="chips">
          <span class="chip">scatter PCA</span>
          <span class="chip">heatmap de z-scores</span>
          <span class="chip">perfil vs promedio</span>
          <span class="chip">tabla interactiva</span>
          <span class="chip">importancias known/cold</span>
        </div>
        <div class="footnote">
          La selección es sincronizada: puedes cambiar el arquetipo desde el desplegable, hacer clic en el scatter o pulsar sobre una fila de la tabla.
        </div>
      </div>
    </section>

    <section class="layout">
      <aside class="sidebar">
        <div class="panel controls">
          <h2>Selección</h2>
          <label for="archetype-select">Arquetipo</label>
          <select id="archetype-select"></select>
          <label for="search-input">Filtrar tabla</label>
          <input id="search-input" type="text" placeholder="Ej. positiva, veterana, archetype_012" />
          <div id="summary-chips" class="chips"></div>
        </div>

        <div class="panel">
          <h2>Ficha del arquetipo</h2>
          <div id="archetype-title" class="sub"></div>
          <div class="detail-grid" id="detail-grid"></div>
        </div>

        <div class="panel">
          <h2>Variables usadas</h2>
          <div class="footnote" id="variables-used"></div>
          <h3 style="margin-top:14px;">Variables descartadas</h3>
          <div class="footnote" id="variables-discarded"></div>
        </div>
      </aside>

      <main class="main">
        <div class="chart-grid">
          <div class="panel">
            <h2>Mapa 2D de arquetipos</h2>
            <div id="scatter-chart" class="chart"></div>
          </div>
          <div class="panel">
            <h2>Perfil del arquetipo seleccionado</h2>
            <div id="profile-chart" class="chart"></div>
          </div>
        </div>

        <div class="panel">
          <h2>Heatmap de z-scores</h2>
          <div id="heatmap-chart" class="wide-chart"></div>
        </div>

        <div class="chart-grid">
          <div class="panel">
            <h2>Top features rama known</h2>
            <div id="known-feature-chart" class="chart"></div>
          </div>
          <div class="panel">
            <h2>Top features rama cold</h2>
            <div id="cold-feature-chart" class="chart"></div>
          </div>
        </div>

        <div class="panel">
          <h2>Tabla de arquetipos</h2>
          <div class="table-wrap">
            <table id="archetype-table">
              <thead>
                <tr>
                  <th>Arquetipo</th>
                  <th>Resumen</th>
                  <th>Usuarios</th>
                  <th>Stars</th>
                  <th>Reviews</th>
                  <th>Tenure</th>
                  <th>Engagement</th>
                  <th>Friends</th>
                  <th>Elite</th>
                  <th>Compliments</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </main>
    </section>
  </div>

  <script>{plotly_js}</script>
  <script>
    const dashboardData = {json.dumps(payload, ensure_ascii=False)};
    const archetypes = dashboardData.archetypes;
    const zscoreRows = dashboardData.zscore_table;
    const metricColumns = dashboardData.metric_columns;
    const displayLabels = dashboardData.display_labels;
    const knownTopFeatures = dashboardData.known_top_features;
    const coldTopFeatures = dashboardData.cold_top_features;
    const featureManifest = dashboardData.feature_manifest;
    const discardedVariables = dashboardData.discarded_variables;

    let selectedId = archetypes[0].user_archetype_id;

    function formatPct(value) {{
      return (value * 100).toFixed(2) + '%';
    }}

    function formatNum(value) {{
      return Number(value).toLocaleString('es-ES', {{ maximumFractionDigits: 3 }});
    }}

    function getArchetype(id) {{
      return archetypes.find(row => row.user_archetype_id === id);
    }}

    function getZScoreRow(id) {{
      return zscoreRows.find(row => row.user_archetype_id === id);
    }}

    function populateSelect() {{
      const select = document.getElementById('archetype-select');
      select.innerHTML = '';
      archetypes
        .slice()
        .sort((a, b) => b.n_users - a.n_users)
        .forEach(row => {{
          const option = document.createElement('option');
          option.value = row.user_archetype_id;
          option.textContent = `${{row.user_archetype_id}} · ${{row.summary_label}} · ${{formatNum(row.n_users)}} users`;
          select.appendChild(option);
        }});
      select.value = selectedId;
      select.addEventListener('change', (event) => {{
        selectedId = event.target.value;
        syncUI();
      }});
    }}

    function renderSidebar() {{
      const row = getArchetype(selectedId);
      document.getElementById('archetype-title').innerHTML =
        `<strong>${{row.user_archetype_id}}</strong><br>${{row.summary_label}}`;

      const chips = document.getElementById('summary-chips');
      chips.innerHTML = '';
      [row.reputation_descriptor, row.activity_descriptor, row.tenure_descriptor, row.social_descriptor, row.prestige_descriptor].forEach(label => {{
        const el = document.createElement('span');
        el.className = 'chip';
        el.textContent = label;
        chips.appendChild(el);
      }});

      const details = [
        ['Usuarios', formatNum(row.n_users)],
        ['Peso relativo', formatPct(row.size_pct)],
        ['Stars medias', formatNum(row.user_average_stars_mean)],
        ['Review count log1p', formatNum(row.user_review_count_log1p_mean)],
        ['Tenure years', formatNum(row.user_tenure_years_mean)],
        ['Engagement', formatNum(row.user_engagement_log1p_mean)],
        ['Friends', formatNum(row.user_friends_log1p_mean)],
        ['Elite years', formatNum(row.user_elite_years_count_mean)],
        ['Compliments', formatNum(row.user_compliment_log1p_total_mean)],
        ['Compliment nz', formatNum(row.user_compliment_nonzero_count_mean)],
      ];

      const grid = document.getElementById('detail-grid');
      grid.innerHTML = details.map(([key, val]) => `
        <div class="detail-card">
          <div class="key">${{key}}</div>
          <div class="val">${{val}}</div>
        </div>
      `).join('');

      document.getElementById('variables-used').innerHTML =
        featureManifest.used_base_variables.slice(0, 8).map(x => `<span class="mono">${{x}}</span>`).join('<br>')
        + '<br><br><strong>Semillas de arquetipo</strong><br>'
        + featureManifest.archetype_seed_variables.map(x => `<span class="mono">${{x}}</span>`).join('<br>');

      document.getElementById('variables-discarded').innerHTML =
        discardedVariables.discarded_feature_families.map(x => `• <span class="mono">${{x}}</span>`).join('<br>');
    }}

    function renderScatter() {{
      const trace = {{
        x: archetypes.map(r => r.pc1),
        y: archetypes.map(r => r.pc2),
        text: archetypes.map(r => `${{r.user_archetype_id}}<br>${{r.summary_label}}<br>Users: ${{formatNum(r.n_users)}}<br>Stars: ${{formatNum(r.user_average_stars_mean)}}`),
        customdata: archetypes.map(r => r.user_archetype_id),
        mode: 'markers+text',
        textposition: 'top center',
        hovertemplate: '%{{text}}<extra></extra>',
        marker: {{
          size: archetypes.map(r => 10 + 28 * Math.sqrt(r.size_pct)),
          color: archetypes.map(r => r.user_average_stars_mean),
          colorscale: 'RdYlGn',
          reversescale: false,
          line: {{
            width: archetypes.map(r => r.user_archetype_id === selectedId ? 3 : 0.8),
            color: archetypes.map(r => r.user_archetype_id === selectedId ? '#0b6e4f' : 'rgba(31,43,33,0.35)')
          }},
          colorbar: {{ title: 'Stars' }},
          opacity: 0.92
        }},
        type: 'scattergl'
      }};
      Plotly.newPlot('scatter-chart', [trace], {{
        margin: {{ l: 40, r: 20, t: 20, b: 40 }},
        xaxis: {{ title: 'PC1' }},
        yaxis: {{ title: 'PC2' }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}, {{ responsive: true }});

      document.getElementById('scatter-chart').on('plotly_click', (event) => {{
        selectedId = event.points[0].customdata;
        document.getElementById('archetype-select').value = selectedId;
        syncUI();
      }});
    }}

    function renderProfile() {{
      const row = getArchetype(selectedId);
      const globalMeans = {{}};
      metricColumns.forEach(col => {{
        globalMeans[col] = archetypes.reduce((acc, cur) => acc + cur[col], 0) / archetypes.length;
      }});
      const selectedValues = metricColumns.map(col => row[col]);
      const globalValues = metricColumns.map(col => globalMeans[col]);

      Plotly.newPlot('profile-chart', [
        {{
          type: 'bar',
          x: metricColumns.map(col => displayLabels[col]),
          y: selectedValues,
          name: row.user_archetype_id,
          marker: {{ color: '#0b6e4f' }},
        }},
        {{
          type: 'scatter',
          mode: 'lines+markers',
          x: metricColumns.map(col => displayLabels[col]),
          y: globalValues,
          name: 'Promedio arquetipos',
          line: {{ color: '#d97706', width: 3 }},
          marker: {{ size: 7 }}
        }}
      ], {{
        margin: {{ l: 50, r: 10, t: 20, b: 90 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        legend: {{ orientation: 'h' }},
      }}, {{ responsive: true }});
    }}

    function renderHeatmap() {{
      const z = metricColumns.map(col => zscoreRows.map(row => row[col]));
      Plotly.newPlot('heatmap-chart', [{{
        type: 'heatmap',
        x: zscoreRows.map(row => row.user_archetype_id),
        y: metricColumns.map(col => displayLabels[col]),
        z: z,
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: 'Metric: %{{y}}<br>Archetype: %{{x}}<br>z-score: %{{z}}<extra></extra>'
      }}], {{
        margin: {{ l: 120, r: 20, t: 20, b: 90 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}, {{ responsive: true }});

      document.getElementById('heatmap-chart').on('plotly_click', (event) => {{
        selectedId = event.points[0].x;
        document.getElementById('archetype-select').value = selectedId;
        syncUI();
      }});
    }}

    function renderFeatureCharts() {{
      Plotly.newPlot('known-feature-chart', [{{
        type: 'bar',
        orientation: 'h',
        x: knownTopFeatures.slice().reverse().map(r => r.gain),
        y: knownTopFeatures.slice().reverse().map(r => r.feature),
        marker: {{ color: '#2563eb' }},
      }}], {{
        margin: {{ l: 180, r: 20, t: 20, b: 40 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}, {{ responsive: true }});

      Plotly.newPlot('cold-feature-chart', [{{
        type: 'bar',
        orientation: 'h',
        x: coldTopFeatures.slice().reverse().map(r => r.gain),
        y: coldTopFeatures.slice().reverse().map(r => r.feature),
        marker: {{ color: '#d97706' }},
      }}], {{
        margin: {{ l: 220, r: 20, t: 20, b: 40 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}, {{ responsive: true }});
    }}

    function renderTable(filterText = '') {{
      const tbody = document.querySelector('#archetype-table tbody');
      const needle = filterText.trim().toLowerCase();
      const rows = archetypes
        .filter(row => !needle || JSON.stringify(row).toLowerCase().includes(needle))
        .sort((a, b) => b.n_users - a.n_users);

      tbody.innerHTML = rows.map(row => `
        <tr data-id="${{row.user_archetype_id}}" class="${{row.user_archetype_id === selectedId ? 'table-row-active' : ''}}">
          <td class="mono">${{row.user_archetype_id}}</td>
          <td>${{row.summary_label}}</td>
          <td>${{formatNum(row.n_users)}}</td>
          <td>${{formatNum(row.user_average_stars_mean)}}</td>
          <td>${{formatNum(row.user_review_count_log1p_mean)}}</td>
          <td>${{formatNum(row.user_tenure_years_mean)}}</td>
          <td>${{formatNum(row.user_engagement_log1p_mean)}}</td>
          <td>${{formatNum(row.user_friends_log1p_mean)}}</td>
          <td>${{formatNum(row.user_elite_years_count_mean)}}</td>
          <td>${{formatNum(row.user_compliment_log1p_total_mean)}}</td>
        </tr>
      `).join('');

      tbody.querySelectorAll('tr').forEach(tr => {{
        tr.addEventListener('click', () => {{
          selectedId = tr.dataset.id;
          document.getElementById('archetype-select').value = selectedId;
          syncUI();
        }});
      }});
    }}

    function syncUI() {{
      renderSidebar();
      renderScatter();
      renderProfile();
      renderHeatmap();
      renderFeatureCharts();
      renderTable(document.getElementById('search-input').value);
    }}

    populateSelect();
    renderFeatureCharts();
    renderHeatmap();
    syncUI();

    document.getElementById('search-input').addEventListener('input', (event) => {{
      renderTable(event.target.value);
    }});
  </script>
</body>
</html>"""

    save_path.write_text(html, encoding="utf-8")
    summary = {
        "artifact_root": str(artifact_root),
        "save_path": str(save_path),
        "n_archetypes": int(len(archetypes)),
        "included_known_features": int(len(known_importance)),
        "included_cold_features": int(len(cold_importance)),
    }
    _save_json(artifact_root / "archetype_dashboard_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
