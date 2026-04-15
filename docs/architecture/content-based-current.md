# Arquitectura Actual De Content-Based

- Proposito: describir la arquitectura realmente implementada hoy en la rama `content-based`.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-13`

## Resumen

La rama `content-based` tiene hoy seis capas reales:

1. representacion estructurada de negocio
2. representacion de usuario, en dos familias:
   - manual
   - deep
3. export de embeddings deep y diagnostico downstream
4. router LightGBM competitivo con ramas para cold start, known users largos y known users prefix-deep
5. router known-user deep como linea experimental fuerte
6. router `two tower + cross + prefix memory` como linea experimental evaluada

## 1. Representacion De Negocio

La base comun de la rama es la representacion de negocio construida por:

- [`build_business_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)
- [`business_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/business_representation.py)

Matrices expuestas:

- `content_matrix`
- `prior_matrix`
- `full_matrix`

Bloques principales:

- `geo`
- `categories`
- `attributes`
- `hours`
- `priors`

## 2. Representacion Manual Del Usuario

La familia manual se construye por:

- [`build_user_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)
- [`user_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/user_representation.py)

Idea principal:

- el perfil del usuario se agrega desde negocios valorados previamente
- la metadata segura vive en un bloque separado

Modos soportados:

- `mean`
- `rating`
- `centered`
- `recency`

Outputs principales:

- `user_profile_features`
- `user_metadata_features`
- `user_full_features`

## 3. Arquitectura Deep Vigente

La arquitectura deep vigente esta implementada en:

- [`deep_user_encoder.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/deep_user_encoder.py)
- [`deep_user_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/deep_user_embeddings.py)

### Bloques Principales

- `business_tower`
- `rating_encoder`
- `history_content_gate` y `history_rating_gate`
- `history_residual_encoder`
- `metadata_encoder` y `base_user_encoder`
- `history_shrinkage_gate`
- `user_fusion`
- `scorer`

### Flujo De `encode_user(...)`

`encode_user(...)` usa solo:

- historial de negocios
- ratings del historial
- mascara del historial
- metadata del usuario

No recibe directamente el negocio candidato.

### Flujo De `forward(...)`

`forward(...)` hace:

1. construir `user_embedding` con `encode_user(...)`
2. construir `candidate_embedding` con `encode_business(...)`
3. combinar:
   - `user_embedding`
   - `candidate_embedding`
   - `abs(user - candidate)`
   - `dot(user, candidate)`
4. pasar todo al `scorer`

### Implicacion Arquitectonica

- El negocio candidato no entra de forma explicita en `encode_user(...)`.
- Pero historial y candidato comparten la misma `business_tower`.
- Por eso la arquitectura sigue parcialmente acoplada entre representacion de historial y representacion de candidato.

## 4. Export De Embeddings

El pipeline deep exporta:

- `user_deep_features.npz`
- `business_deep_features.npz`
- `user_deep_summary.json`
- `deep_user_encoder_checkpoint.pt`

La exportacion de usuarios se hace con historial completo disponible en train y metadata segura.

El snapshot oficial vigente para export es `competition_embeddings_v3_iter03`.

## 5. Downstream Real Sobre Embeddings

Existen dos usos downstream ya implementados:

- [`train_frozen_embedding_regressor.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_regressor.py)
- [`train_lgbm_raw_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_raw_router.py)

El frozen regressor:

- toma `user_deep_features` y `business_deep_features`
- anade `review_context`
- entrena cabezas Ridge y MLP downstream
- compara runs y selecciona el mejor experimento

El router prefix-deep:

- conserva `raw_core` para usuarios conocidos largos
- usa arquetipos para cold start
- reutiliza el bundle deep exportado para representar prefijos de usuarios conocidos
- activa la rama prefix-deep solo donde la validacion lo justifica

## 6. Arquitectura Router Vigente

La politica actual es:

- `history_band = 0` -> `cold_model`
- `history_band = 6-20` -> `known_prefix_deep_model`
- otros usuarios conocidos -> `known_model`

### Feature `user_average_stars` — derivacion actual

Hasta `lgbm_raw_router_prefix_deep_v1`, `user_average_stars` se tomaba directamente de `usuarios.csv` (el agregado all-time de Yelp). Ese agregado puede incluir reviews del periodo de test para usuarios activos, lo que constituye leakage potencial — especialmente critico en cold start, donde era la feature mas importante (gain 9.1×10^7 en el modelo cold, 4× sobre la siguiente).

A partir de `lgbm_train_stars_v1`:

- `user_average_stars` se reemplaza por la media de estrellas calculada exclusivamente desde `train_reviews` (o `train_split` en la rama de validacion)
- usuarios sin historial en train reciben la media global como fallback
- el calculo esta encapsulado en `build_train_user_stars(train_reviews_df, global_mean)` en `utils/lgbm_raw_features.py`
- el reemplazo ocurre en `train_lgbm_raw_router.py` antes de pasar `users_df` a cualquier builder de features; se propaga automaticamente a `user_average_stars`, `user_minus_global_mean` y `user_business_metadata_gap`
- la rama de submission usa `users_df_sub` con la media calculada sobre el conjunto de train completo

Lo que sigue sin formar parte de la arquitectura actual:

- `interaction-first`
- doble torre separada para historial y candidato
- GNN global usuario-negocio
- una sola submission deep como camino principal de competencia

## 7. Arquitectura `known_user_deep_e2e`

La linea experimental actual esta implementada en:

- [`known_user_deep_e2e.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/known_user_deep_e2e.py)
- [`utils/known_user_deep_e2e.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/known_user_deep_e2e.py)

Bloques principales:

- `business_tower`
- `event_encoder`
- `user_type_encoder`
- `history_attention`
- `positive_attention`
- `negative_attention`
- `band_1_taste_fusion`
- `band_2_3_taste_fusion`
- `band_4_5_taste_fusion`
- `band_6_20_taste_fusion`
- `band_gt_20_taste_fusion`
- `baseline_head`
- `band_1_gate_head`
- `band_2_3_gate_head`
- `band_4_5_gate_head`
- `band_6_20_gate_head`
- `band_gt_20_gate_head`
- `band_1_correction_head`
- `band_2_3_correction_head`
- `band_4_5_correction_head`
- `band_6_20_correction_head`
- `band_gt_20_correction_head`

Caracteristicas importantes de la version actual:

- usa `business_content_features` del bundle oficial como entrada densa
- consume prefijos temporales por fila, no embeddings de usuario fijos exportados
- usa un trunk compartido y cinco expertos internos por banda efectiva de historial:
  - `1`
  - `2-3`
  - `4-5`
  - `6-20`
  - `>20`
- mantiene el reporting y la activacion externa del router en bandas gruesas:
  - `1`
  - `2-5`
  - `6-20`
  - `>20`
- el split interno `2-3` / `4-5` se decide con el `history_count` exacto derivado del prefijo disponible
- recibe `incumbent_prediction_raw` como entrada del dataset preparado
- ya no intenta sustituir conceptualmente al incumbent con una prediccion libre
- aprende una correccion acotada sobre el incumbent
- normaliza `baseline_features`, `user_numeric_features`, `user_aux_features` y `event_scalar_features`
- mantiene `baseline_hat` como senal auxiliar estable
- acota la correccion con `tanh` para evitar sobrecorreccion
- el incumbent y el dataset deep reconstruyen `history_band` con la misma logica basada en `context_reviews`, corrigiendo la inconsistencia historica de la banda `1`
- el summary de validacion ahora persiste metricas explicitas para:
  - `2`
  - `3`
  - `4`
  - `5`
  - `2-3`
  - `4-5`

Estado:

- experimental
- candidato fuerte
- mejor snapshot estable actual en [`known_user_deep_router_v2_eval_v3`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v3)
- snapshot `v4` de diagnostico en [`known_user_deep_router_v4_eval_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v4_eval_v1)
- bandas activadas en el mejor rerun estable:
  - `1`
  - `2-5`
  - `6-20`
  - `>20`
- lectura actual:
  - `v3` sigue siendo mejor que `v4` en MAE global
  - `v4` fue util para comprobar que `4-5` responde mejor que `2-3`
  - el split interno corto no se ha promovido todavia como nueva base oficial

## 8. Arquitectura `known_user_two_tower_cross`

La linea `two tower` esta implementada en:

- [`known_user_two_tower_cross.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/known_user_two_tower_cross.py)
- [`utils/known_user_two_tower_cross.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/known_user_two_tower_cross.py)
- [`train_known_user_two_tower_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_known_user_two_tower_router.py)

Bloques principales:

- `business_tower`
- `event_encoder`
- `user_context_encoder`
- `query_projection`
- `history_attention`
- memorias de prefijo:
  - `prefix_mean`
  - `prefix_recency`
  - `prefix_attention`
  - `positive`
  - `negative`
  - `last_event`
- `user_fusion`
- `baseline_head`
- `cross_layers`
- `cross_projection`
- `correction_head`
- `alpha_head`

Caracteristicas importantes de la version evaluada:

- usa una torre compartida para negocio candidato e historial
- construye varias memorias de prefijo antes del bloque `cross`
- aprende una correccion residual sobre `incumbent_prediction_raw`
- evalua la rama solo para `history_band in {1, 2-5, 6-20, >20}`
- usa `structured_from_scratch` como fuente de negocio en el mejor run auditado

Estado real tras el entrenamiento `known_user_two_tower_router_v2_eval_v2`:

- experimental
- no recomendado para submission
- ninguna banda activada en validacion
- `best_val_mae = 0.6890`
- la rama deep tuvo cobertura total en usuarios conocidos, pero empeoro el MAE en todas las bandas

Lectura arquitectonica:

- el `two tower` actual parece demasiado agresivo como corrector del incumbent
- hay señal para mejorar RMSE, pero no para optimizar el criterio competitivo principal de MAE
- hoy no desplaza ni al `known_model` ni al `known_user_deep_e2e`
