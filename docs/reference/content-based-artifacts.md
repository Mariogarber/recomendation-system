# Artefactos De Content-Based

- Proposito: definir el contrato principal de artefactos de la rama `content-based`.
- Tipo documental: `reference`
- Ultima actualizacion: `2026-04-11`

## Familias Principales

### Business Representation

Ubicacion tipica:

- `content-based/artifacts/business_repr_*`

Ficheros clave:

- `business_ids.csv`
- `business_content_features.npz`
- `business_prior_features.npz`
- `business_full_features.npz`
- `business_feature_names.json`
- `feature_metadata.csv`
- `business_representation_summary.json`

### Manual User Representation

Ubicacion tipica:

- `content-based/artifacts/user_repr_*`

Ficheros clave:

- `user_ids.csv`
- `user_profile_features.npz`
- `user_metadata_features.npz`
- `user_full_features.npz`
- `user_feature_names.json`
- `user_feature_metadata.csv`
- `user_profile_summary.json`

### Deep Competition Embeddings

Ubicacion tipica:

- `content-based/artifacts/competition_embeddings_*`

Subarboles:

- `business_repr/`
- `user_manual_repr/`
- `user_deep_repr/`

En `user_deep_repr/`:

- `user_deep_ids.csv`
- `business_deep_ids.csv`
- `user_deep_features.npz`
- `business_deep_features.npz`
- `user_deep_feature_names.json`
- `user_deep_feature_metadata.csv`
- `business_deep_feature_metadata.csv`
- `user_deep_summary.json`
- `deep_user_encoder_checkpoint.pt`

### Embedding Report

Ubicacion tipica:

- `content-based/artifacts/*/report_*`

Outputs tipicos:

- HTML de reporte
- CSVs de diagnostico
- JSONs resumen

### Frozen Regressor

Ubicacion tipica:

- `content-based/artifacts/frozen_embedding_regressor_*`

Ficheros clave:

- `split_summary.json`
- `validation_summary.json`
- `band_metrics.csv`
- `validation_predictions.csv`
- `experiment_ranking.csv`
- `run_summary.json`

### Routed LGBM Router

Ubicacion tipica:

- `content-based/artifacts/lgbm_raw_router_*`

Ficheros clave del snapshot prefix-deep actual:

- `training_summary.json`
- `validation_summary.json`
- `validation_predictions.csv`
- `submission.csv`
- `feature_manifest.json`
- `discarded_variables.json`
- `validation_router_spec.joblib`
- `submission_router_spec.joblib`
- `known_validation_model.txt`
- `known_prefix_validation_model.txt`
- `cold_validation_model.txt`
- `known_submission_model.txt`
- `known_prefix_submission_model.txt`
- `cold_submission_model.txt`
- `archetype_profiles.csv`

Snapshot recomendado hoy:

- [`lgbm_raw_router_prefix_deep_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_prefix_deep_v1)

Snapshot historico de referencia:

- [`lgbm_raw_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1)

## Regla De Alineacion

- `user_id` es la clave de alineacion para artefactos de usuario
- `business_id` es la clave de alineacion para artefactos de negocio
- cualquier cambio en el orden o cobertura de ids obliga a regenerar matrices dependientes

## Fuente Oficial De Recomendaciones

Los artefactos recomendados no se declaran aqui. Se declaran en:

- [Registro De Experimentos](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/registry.md)
