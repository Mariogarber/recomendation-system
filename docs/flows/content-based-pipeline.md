# Pipeline Operativo De Content-Based

- Proposito: describir el flujo real, ejecutable y actual de la rama `content-based`.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-10`

## Vista End-To-End

El pipeline actual de `content-based` tiene seis etapas operativas:

1. auditoria de dataset y leakage
2. construccion de representacion de negocio
3. construccion de representacion manual de usuario
4. construccion de embeddings de competicion
5. reporte diagnostico de embeddings
6. regressor downstream sobre embeddings congelados

## 1. Auditoria

Script:

- [`phase1_audit.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/phase1_audit.py)

Objetivo:

- resumir dataset
- medir cold start
- auditar leakage de metadata
- resumir cobertura de parsers

## 2. Business Representation

Script:

- [`build_business_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)

Output contractual principal:

- `business_ids.csv`
- `business_content_features.npz`
- `business_prior_features.npz`
- `business_full_features.npz`
- `business_feature_names.json`
- `business_block_summary.csv`
- `feature_metadata.csv`

## 3. Manual User Representation

Script:

- [`build_user_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)

Output contractual principal:

- `user_ids.csv`
- `user_profile_features.npz`
- `user_metadata_features.npz`
- `user_full_features.npz`
- `user_feature_names.json`
- `user_feature_metadata.csv`

## 4. Competition Embeddings

Script:

- [`build_competition_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_competition_embeddings.py)

Subarboles generados:

- `business_repr/`
- `user_manual_repr/`
- `user_deep_repr/`

Outputs clave del deep bundle:

- `user_deep_features.npz`
- `business_deep_features.npz`
- `user_deep_summary.json`
- `deep_user_encoder_checkpoint.pt`

## 5. Embedding Report

Script:

- [`analyze_embeddings_report.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/analyze_embeddings_report.py)

Este reporte sirve para:

- coverage y health
- utilidad
- coherencia de negocio
- consistencia de usuario
- clustering
- homofilia social

Importante:

- es diagnostico
- no sustituye al protocolo formal de evaluacion y seleccion downstream

## 6. Frozen Regressor Downstream

Script:

- [`train_frozen_embedding_regressor.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_regressor.py)

Salidas tipicas:

- `split_summary.json`
- `validation_summary.json`
- `band_metrics.csv`
- `validation_predictions.csv`
- `experiment_ranking.csv`
- `run_summary.json`

## Flujo Recomendado Hoy

1. ejecutar auditoria
2. construir o refrescar embeddings de competicion
3. ejecutar reporte diagnostico del bundle exportado
4. ejecutar frozen regressor para evaluacion downstream
5. registrar snapshot y veredicto en `docs/experiments/registry.md`
