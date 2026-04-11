# Raw Router Archetypes Experiment

Fecha: `2026-04-11`

## Objetivo

Implementar la fase B del plan de mejora sobre `raw_core`:

- conservar el camino ganador para usuarios conocidos
- enriquecer solo la rama de cold start con una representacion de usuario basada en arquetipos
- exportar una submission en formato de competicion

## Proceso implementado

1. Se tomo `raw_core` como rama `known_model`.
2. Se construyo una representacion metadata-only del usuario usando:
   - estrellas medias
   - intensidad de reviews
   - antiguedad
   - votos e interaccion
   - amigos
   - elite
   - compliments
3. Se agruparon usuarios en `64` arquetipos con `MiniBatchKMeans`.
4. Se generaron afinidades train-only entre arquetipo y facets del negocio:
   - media global del arquetipo
   - estado
   - ciudad top
   - bin de estrellas del negocio
   - abierto/cerrado
   - familia principal de categoria
5. Se entreno un `cold_model` que usa:
   - todas las features de `raw_core`
   - variables de arquetipo
   - gaps arquetipo-negocio
6. Se exporto un router duro:
   - usuario conocido -> `known_model`
   - usuario nuevo -> `cold_model`

## Resultado

Artefacto principal:

- [`content-based/artifacts/lgbm_raw_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1)

Metricas locales:

- `raw_core` previo:
  - `validation_mae_rounded = 0.6204`
  - `band_0 mae = 0.5798`
- `raw_router_v1`:
  - `validation_mae_rounded = 0.6269`
  - `band_0 mae = 0.5896`

Lectura:

- el router enriquecido ya funciona end-to-end
- la rama cold aprendio senal util y sus features dominantes son coherentes
- en esta iteracion no supera todavia al `raw_core` puro en la validacion local
- aun asi queda exportada la submission experimental de fase B para comparar en leaderboard

## Variables mas influyentes

Rama `known_model`:

- `user_average_stars`
- `business_stars`
- `user_minus_global_mean`
- `user_business_metadata_gap`
- `business_minus_global_mean`
- `user_review_count`
- `user_engagement_log1p`
- `user_total_votes`

Rama `cold_model`:

- `user_average_stars`
- `archetype_star_bin_mean`
- `user_minus_global_mean`
- `business_stars`
- `user_archetype_id`
- `archetype_train_mean`
- `archetype_open_mean`
- `archetype_business_star_gap`
- `business_primary_category_family`
- `business_city_top`

Referencias:

- [`known_feature_importance.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/known_feature_importance.csv)
- [`cold_feature_importance.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/cold_feature_importance.csv)

## Arquetipos

Todos los arquetipos aprendidos, con su tamano y medias de metadata, estan documentados en:

- [`archetype_profiles.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/archetype_profiles.csv)
- [`archetype_dashboard.html`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/archetype_dashboard.html)

Ejemplos representativos:

- `archetype_006`
  - usuarios con `average_stars` muy alta, poco volumen de reviews y baja interaccion
- `archetype_016`
  - usuarios de rating muy bajo, poca actividad y muy poco capital social
- `archetype_012`
  - usuarios veteranos, hiperactivos, con muchos votos, muchos compliments y senal fuerte de elite
- `archetype_005`
  - usuarios recientes, poco activos y con engagement casi nulo

## Variables usadas y descartadas

Documentacion completa:

- [`feature_manifest.json`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/feature_manifest.json)
- [`discarded_variables.json`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/discarded_variables.json)

Resumen de descartes:

- no se usan ids directos
- no se usan priors por `user_id` o `business_id` del experimento `raw_priors`
- no se usan embeddings deep en esta fase
- no se usan campos de texto libre como features directas

## Deliverable

Submission experimental de fase B:

- [`submission.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1/submission.csv)

Resumen de export:

- `414765` filas
- `244828` filas por rama `known`
- `169937` filas por rama `cold`
