# Estado Actual Del Repositorio

- Proposito: resumir el estado vigente del repositorio y centralizar que esta implementado, que esta pendiente y que snapshots son los recomendados hoy.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-11`

## Objetivo Del Proyecto

El repositorio sigue centrado en rating prediction para variantes de un dataset derivado de Yelp.

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
- routed LightGBM stack con `raw_core`, arquetipos para cold start y rama prefix-deep para usuarios conocidos intermedios
- router deep de usuarios conocidos como linea experimental adicional

### Pendiente O No Cerrado

- mejorar la banda `2-5`, que sigue siendo la zona mas debil en el router actual aunque la rama prefix-deep ya activa `6-20`
- seguir refinando representaciones de usuario para bandas cortas
- cerrar si la rama `known_user_deep_e2e` puede competir de verdad tras la correccion de normalizacion y salida acotada
- consolidar una unica narrativa de submission para que los hubs apunten siempre al snapshot oficial
- trazabilidad mas rica para artefactos legacy en `models/` y salidas en `data/`

## Estado De `content-based`

### Etapas Ejecutables Reales

- auditoria: [`phase1_audit.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/phase1_audit.py)
- representacion de negocio: [`build_business_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)
- representacion manual de usuario: [`build_user_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)
- embeddings de competicion: [`build_competition_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_competition_embeddings.py)
- reporte diagnostico: [`analyze_embeddings_report.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/analyze_embeddings_report.py)
- regressor downstream: [`train_frozen_embedding_regressor.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_regressor.py)
- router LightGBM competitivo: [`train_lgbm_raw_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_raw_router.py)
- router known-user deep candidato: [`train_known_user_deep_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_known_user_deep_router.py)

### Situacion Arquitectonica

- La arquitectura deep vigente sigue usando una `business_tower` compartida para historial y candidato, un `encode_user(...)` construido desde historial y metadata, y un `scorer` final con el negocio candidato.
- Esa arquitectura esta implementada en [`deep_user_encoder.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/deep_user_encoder.py).
- El uso mas fuerte del bundle deep hoy no es un scorer end-to-end nuevo, sino su reutilizacion como fuente de embeddings exportados y como entrada del router prefix-deep.
- La linea `known_user_deep_e2e` existe como experimento entrenable, pero no esta activada en el snapshot candidato visible.
- El codigo de esa linea ya incorpora normalizacion de bloques tabulares y una salida acotada alrededor de la media global para evitar el colapso observado en runs previos.

## Estado De `collaborative-filtering`

- Es la parte mas madura del repo.
- Contiene baseline models, KNN, PMF, BPMF, NNBPMF, ensembles y utilidades de analisis/prediccion.
- Su documentacion vigente vive en `docs/reference/` y `docs/flows/`.

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
- snapshot oficial de router con rama prefix-deep activada solo para `6-20`:
  - [`lgbm_raw_router_prefix_deep_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_prefix_deep_v1)
- snapshot candidato de router known-user deep:
  - [`known_user_deep_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v1)
- snapshot candidato o de referencia para comparacion de cabeza de entrenamiento:
  - [`competition_embeddings_v3_iter04`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04)
- run oficial downstream disponible:
  - [`frozen_embedding_regressor_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_v1)
- baseline router previo, ahora referencia historica:
  - [`lgbm_raw_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1)

## Proximos Pasos Reales

- seguir usando `docs/` como unica fuente de verdad para el estado actual
- registrar en el decision log cualquier nuevo cambio sobre la arquitectura deep o el router
- si se vuelve a tocar la rama known-user, priorizar primero la banda `2-5`
- mantener los snapshots oficiales y candidatos sincronizados con `docs/experiments/registry.md`
