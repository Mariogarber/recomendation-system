# Registro Oficial De Runs Y Snapshots

- Proposito: declarar el estado oficial de snapshots, runs y bundles recomendados del repositorio.
- Tipo documental: `experiment`
- Ultima actualizacion: `2026-04-13`

## Regla

Este es el unico documento que declara que snapshot es `official`, `candidate` o `deprecated`.

## Content-Based

| Activo | Estado | Uso principal | Notas |
|---|---|---|---|
| [`competition_embeddings_v3_iter03`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter03) | `official` | export de embeddings deep | bundle recomendado actual para export |
| [`competition_embeddings_v3_iter04`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04) | `candidate` | comparacion de training head | referencia adicional usada en downstream |
| [`frozen_embedding_regressor_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_v1) | `official` | evaluacion downstream | run formal disponible |
| [`lgbm_raw_router_prefix_deep_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_prefix_deep_v1) | `official` | submission router | router actual con cold, known y known-prefix-deep |
| [`known_user_deep_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v1) | `candidate` | router known-user deep | linea experimental, sin bandas activadas en el snapshot visible |
| [`known_user_deep_router_v2_eval_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v2) | `candidate` | router known-user deep | mejor linea deep conocida; activa `1`, `2-5`, `6-20` y `>20` |
| [`known_user_two_tower_router_v2_eval_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_two_tower_router_v2_eval_v2) | `deprecated` | router known-user two-tower | evaluado con GPU; sin bandas activadas y sin mejora operativa sobre el incumbent |
| [`lgbm_raw_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1) | `candidate` | baseline router previo | referencia historica inmediata para comparacion |
| [`meta_lgbm_hybrid_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v1) | `official` | mejor submission conocido | meta-LightGBM CF bias + CB deep router; leaderboard 0.6529; val MAE known 0.665 (vs CB raw 0.711); comparacion correcta vs CB rounded 0.669 delta -0.004 |
| [`meta_lgbm_hybrid_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v2) | `deprecated` | experimento CF+CB con fix | Direction 2 fix + user/item bias features; CF MAE 1.004 arrastra el modelo; val MAE 0.687 > baseline 0.669; no mejorar |
| [`meta_lgbm_hybrid_v3`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v3) | `candidate` | CB + bias features sin CF | Sin CF; val MAE 0.673 vs CB rounded 0.669; delta +0.003 — meta no aporta sobre CB solo |
| [`competition_embeddings_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v1) | `deprecated` | historico | version temprana |
| [`competition_embeddings_v2_smoke`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v2_smoke) | `deprecated` | smoke test | no usar como referencia oficial |
| [`competition_embeddings_v3_iter01`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter01) | `deprecated` | historico | iteracion superada |
| [`competition_embeddings_v3_iter02`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter02) | `deprecated` | historico | iteracion superada |

## Convencion

- `official`: snapshot recomendado hoy
- `candidate`: snapshot prometedor pero no consolidado como oficial
- `deprecated`: snapshot historico, smoke o superado
