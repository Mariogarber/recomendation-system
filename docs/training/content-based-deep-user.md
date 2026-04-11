# Training Deep User De Content-Based

- Proposito: fijar el protocolo estable de entrenamiento de la familia deep actual.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-10`

## Objetivo

Entrenar una familia de embeddings profundos de usuario y negocio para rating prediction a partir de:

- `business_full_features`
- historial de interacciones del usuario
- ratings del historial
- metadata segura del usuario

## Scripts Y Codigo Relevante

- builder y entrenamiento:
  - [`build_competition_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_competition_embeddings.py)
- implementacion principal:
  - [`deep_user_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/deep_user_embeddings.py)
- modelo:
  - [`deep_user_encoder.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/deep_user_encoder.py)

## Protocolo Estable

- construir arrays prefix-safe para train
- usar validacion temporal
- optimizar regresion de rating con `SmoothL1Loss`
- seleccionar mejor epoch por `val_mae`
- reentrenar el modelo final sobre todo el train con el numero de epochs seleccionado
- exportar embeddings de usuario y negocio

## Metricas

- primaria: `MAE`
- secundaria: `RMSE`

## Artefactos Generados

- `user_deep_ids.csv`
- `business_deep_ids.csv`
- `user_deep_features.npz`
- `business_deep_features.npz`
- `user_deep_feature_names.json`
- `user_deep_summary.json`
- `deep_user_encoder_checkpoint.pt`

## Snapshots Relevantes

- oficial actual de export:
  - [`competition_embeddings_v3_iter03`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter03)
- candidato de comparacion:
  - [`competition_embeddings_v3_iter04`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04)

## Lo Que No Debe Mezclarse Aqui

Este documento no es:

- un log de iteraciones historicas
- un RFC de una arquitectura nueva
- el contrato completo de artefactos

Para eso usar:

- [Log Deep User](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/content-based-deep-user-log.md)
- [Decision Log](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/decision-log.md)
- [Artefactos De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/content-based-artifacts.md)
