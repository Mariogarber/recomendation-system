# Arquitectura Actual De Content-Based

- Proposito: describir la arquitectura realmente implementada hoy en la rama `content-based`.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-11`

## Resumen

La rama `content-based` tiene hoy cinco capas reales:

1. representacion estructurada de negocio
2. representacion de usuario, en dos familias:
   - manual
   - deep
3. export de embeddings deep y diagnostico downstream
4. router LightGBM competitivo con ramas para cold start, known users largos y known users prefix-deep
5. router known-user deep como linea experimental

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
- `taste_fusion`
- `baseline_head`
- `gate_head`
- `residual_head`

Caracteristicas importantes de la version actual:

- usa `business_content_features` del bundle oficial como entrada densa
- consume prefijos temporales por fila, no embeddings de usuario fijos exportados
- normaliza `baseline_features`, `user_numeric_features`, `user_aux_features` y `event_scalar_features`
- acota la base y el residual con `tanh` para evitar saturacion

Estado:

- experimental
- no oficial
- sin bandas activadas en `known_user_deep_router_v1`
