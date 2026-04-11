# Estado Actual Del Repositorio

- Proposito: resumir el estado vigente del repositorio y centralizar que esta implementado, que esta pendiente y que snapshots son los recomendados hoy.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-10`

## Objetivo Del Proyecto

El repositorio esta centrado en rating prediction para variantes de un dataset derivado de Yelp.

Hoy conviven dos lineas principales:

- una rama legacy de collaborative filtering ya bastante madura
- una rama activa de content-based, que es donde esta el trabajo de arquitectura y evaluacion mas reciente

## Estado Global

### Implementado

- modelos, metricas, ensembles y utilidades de collaborative filtering
- builders manuales de representacion de negocio y usuario para content-based
- pipeline deep de embeddings de usuario y negocio para content-based
- auditoria de leakage y cold start para content-based
- reporte diagnostico de embeddings
- regressor downstream sobre embeddings congelados

### Pendiente O No Cerrado

- suite formal y estable de baselines leak-safe para la rama content-based
- politica final de cold start como pieza de prediccion de produccion
- consolidacion final de arquitectura para la siguiente iteracion deep
- trazabilidad mas rica para artefactos legacy en `models/` y salidas en `data/`

## Estado De `content-based`

### Etapas Ejecutables Reales

- auditoria: [`phase1_audit.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/phase1_audit.py)
- representacion de negocio: [`build_business_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)
- representacion manual de usuario: [`build_user_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)
- embeddings de competicion: [`build_competition_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_competition_embeddings.py)
- reporte diagnostico: [`analyze_embeddings_report.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/analyze_embeddings_report.py)
- regressor downstream: [`train_frozen_embedding_regressor.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_regressor.py)

### Situacion Arquitectonica

- La arquitectura deep vigente usa una `business_tower` compartida para historial y candidato, un `encode_user(...)` construido desde historial y metadata, y un `scorer` final con el negocio candidato.
- Esa arquitectura esta implementada en [`deep_user_encoder.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/deep_user_encoder.py).
- Existe una propuesta nueva, aun no implementada, para separar mejor historial y negocio candidato. Esa propuesta vive en el decision log y en el arbol de propuestas.

## Estado De `colaborative-filtering`

- Es la parte mas madura del repo.
- Contiene baseline models, KNN, PMF, BPMF, NNBPMF, ensembles y utilidades de analisis/prediccion.
- Su documentacion vigente vive ahora en `docs/reference/` y `docs/flows/`.

## Hallazgos De Dataset Ya Asumidos

Para la rama content-based, la documentacion vigente mantiene como hechos ya establecidos:

- dataset muy disperso
- cold start dominado por `new_user_known_item`
- riesgo real de leakage en agregados directos de metadata
- alta cobertura y valor predictivo de la metadata de negocio

Los detalles y cifras operativas estan en:

- [Datasets Y Activos De Datos](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/data-assets.md)

## Snapshots Y Artefactos Recomendados Hoy

Las recomendaciones oficiales se declaran solo aqui y en el registro de experimentos.

### Content-Based

- snapshot oficial exportado de deep embeddings:
  - [`competition_embeddings_v3_iter03`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter03)
- snapshot candidato o de referencia para comparacion de cabeza de entrenamiento:
  - [`competition_embeddings_v3_iter04`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04)
- run oficial downstream disponible:
  - [`frozen_embedding_regressor_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_v1)

## Proximos Pasos Reales

- mantener esta nueva arquitectura documental como unica fuente de verdad
- registrar en el decision log cualquier cambio sobre la arquitectura deep
- cuando se implemente el desacoplamiento historial/candidato, actualizar:
  - arquitectura actual
  - flujos
  - training deep
  - contrato de artefactos
  - registro de experimentos
