# Registro Oficial De Runs Y Snapshots

- Proposito: declarar el estado oficial de snapshots, runs y bundles recomendados del repositorio.
- Tipo documental: `experiment`
- Ultima actualizacion: `2026-04-10`

## Regla

Este es el unico documento que declara que snapshot es `official`, `candidate` o `deprecated`.

## Content-Based

| Activo | Estado | Uso principal | Notas |
|---|---|---|---|
| [`competition_embeddings_v3_iter03`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter03) | `official` | export de embeddings deep | bundle recomendado actual para export |
| [`competition_embeddings_v3_iter04`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04) | `candidate` | comparacion de training head | referencia adicional usada en downstream |
| [`frozen_embedding_regressor_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_v1) | `official` | evaluacion downstream | run formal disponible |
| [`competition_embeddings_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v1) | `deprecated` | historico | version temprana |
| [`competition_embeddings_v2_smoke`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v2_smoke) | `deprecated` | smoke test | no usar como referencia oficial |
| [`competition_embeddings_v3_iter01`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter01) | `deprecated` | historico | iteracion superada |
| [`competition_embeddings_v3_iter02`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter02) | `deprecated` | historico | iteracion superada |

## Convencion

- `official`: snapshot recomendado hoy
- `candidate`: snapshot prometedor pero no consolidado como oficial
- `deprecated`: snapshot historico, smoke o superado
