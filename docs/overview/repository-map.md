# Mapa Del Repositorio

- Proposito: describir la estructura global del repositorio y el papel de cada area principal.
- Tipo documental: `reference`
- Ultima actualizacion: `2026-04-10`

## Vision General

El repositorio combina una rama madura de collaborative filtering con una rama activa de content-based y un conjunto de activos legacy de datos, notebooks y modelos serializados.

## Directorios Principales

- [`colaborative-filtering`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering)
  - implementacion legacy y madura de collaborative filtering
  - incluye modelos, metricas, ensembles, utilidades y notebooks historicos
- [`content-based`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based)
  - rama activa para rating prediction basada en contenido
  - incluye auditoria, builders manuales, embeddings deep, reportes y regressor downstream
- [`data`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/data)
  - dataset reducido y salidas legacy de prediccion para la rama collaborative filtering
- [`models`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/models)
  - artefactos serializados legacy de modelos entrenados
- [`docs`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs)
  - fuente de verdad canónica de la documentacion del repo

## Subestructura Relevante

### `colaborative-filtering/`

- `model/`: baseline, KNN, PMF, SVD, BPMF y NNBPMF
- `metric/`: NDCG y similitud usuario-usuario
- `ensemble/`: ensembles clasicos y cold-start adaptive
- `utils/`: analisis, prediccion y splits por bandas
- notebooks principales:
  - `analysis.ipynb`
  - `hiperparameter.ipynb`
  - `knn.ipynb`
  - `notebook_svd_knn.ipynb`
  - `recomendation-system.ipynb`

### `content-based/`

- scripts ejecutables:
  - `phase1_audit.py`
  - `build_business_representation.py`
  - `build_user_representation.py`
  - `build_competition_embeddings.py`
  - `analyze_embeddings_report.py`
  - `train_frozen_embedding_regressor.py`
- `model/`: base, deep user encoder y frozen regressor
- `utils/`: IO, leakage audit, split, builders y joins downstream
- `artifacts/`: smoke tests, iteraciones deep y runs downstream

## Donde Encontrar Cada Tipo De Informacion

- estado actual: [current-state.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/status/current-state.md)
- arquitectura content-based: [content-based-current.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/content-based-current.md)
- flujos operativos: [content-based-pipeline.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/flows/content-based-pipeline.md) y [collaborative-filtering-workflow.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/flows/collaborative-filtering-workflow.md)
- datasets y activos: [data-assets.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/data-assets.md)
- artefactos y snapshots: [content-based-artifacts.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/content-based-artifacts.md), [model-artifacts.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/model-artifacts.md) y [registry.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/registry.md)
