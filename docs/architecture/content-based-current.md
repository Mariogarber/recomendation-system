# Arquitectura Actual De Content-Based

- Proposito: describir la arquitectura realmente implementada hoy en la rama `content-based`.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-10`

## Resumen

La rama `content-based` tiene hoy tres capas reales:

1. representacion estructurada de negocio
2. representacion de usuario, en dos familias:
   - manual
   - deep
3. scoring downstream:
   - diagnostico de embeddings
   - regressor sobre embeddings congelados

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

## 5. Downstream Real Sobre Embeddings

Existe una etapa downstream ya implementada:

- [`frozen_embedding_regressor.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/frozen_embedding_regressor.py)
- [`train_frozen_embedding_regressor.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_regressor.py)

Esta etapa:

- toma `user_deep_features` y `business_deep_features`
- anade `review_context`
- entrena cabezas Ridge y MLP downstream
- compara runs y selecciona el mejor experimento

## 6. Lo Que No Forma Parte De La Arquitectura Actual

No se considera arquitectura vigente:

- `interaction-first`
- doble torre separada para historial y candidato
- GNN global usuario-negocio
- politica final de cold start en inferencia de produccion
