# Estado Actual Del Repositorio

- Proposito: resumir el estado vigente del repositorio y centralizar que esta implementado, que esta pendiente y que snapshots son los recomendados hoy.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-15` (sesion Dir C y D — loss function fix)

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
- router deep de usuarios conocidos como linea experimental fuerte y ya competitiva
- router `two tower + cross + prefix memory` como linea experimental evaluada y no promovida
- tanda `v4` del router deep con split interno `2-3` / `4-5` completada y analizada
- tanda `feature-first` sobre el router LightGBM de corto historial completada con entrenamiento en GPU y ya evaluada contra `v3`
- ciclo completo de meta-stacking (v1-v6) ejecutado y cerrado: ningun meta supero LB 0.6528 de v1; causa raiz identificada (cold rows 41% no corregibles desde meta-corrector externo)
- ciclo de mejora cold model (lgbm_router_v2 a v5) ejecutado y cerrado: ninguna mejora sustancial encontrada; biz_train_stats features confirmadas como daninas; baseline lgbm_train_stars_v1 sigue siendo el mejor cold model
- **Direction B completada (validacion)**: `lgbm_router_v6` — cold band 0 MAE = 1.1315 (-0.0273 vs baseline 1.1588); mejor cold hasta la fecha; run incompleto (no submission artifacts); pendiente re-run completo
- **Direction A cerrada (regresion)**: `known_user_deep_router_v5_direct_v1` — deep_mae=0.6810 (+0.0116 vs v3); best_epoch=1; alpha gate confirmado como estabilizador; smooth_l1 como causa raiz del overfitting
- **Direction C cerrada (regresion)**: `known_user_deep_router_v6_regularized` — C1 (direct+L2) deep_mae=0.6782; C2 (gated wider) deep_mae=0.6750; ambas peores que v2_eval_v3 (0.6694)
- **Direction D1 cerrada**: `known_user_deep_router_v7_mae_v1` — cambio de smooth_l1 a l1_loss (MAE alineado); deep_mae=0.6724; curva monotona confirmada (best_epoch=6); lr=8e-4 demasiado alto para MAE loss; LB 0.6538
- **Direction D2 en curso**: `known_user_deep_router_v7_mae_v2` — lr=3e-4, patience=10, max_epochs=40, correction_scales exactas de v3; run activo

### Pendiente O No Cerrado

- **ciclo meta-stacking cerrado**: v1-v6 ejecutados; mejor LB sigue siendo v1 (0.6528); documentado en `docs/experiments/meta-stacking-experiments-2026-04-14.md`
- **ciclo cold model cerrado**: lgbm_router_v2 a v5 ejecutados; ninguna mejora sustancial; documentado en `docs/experiments/cold-model-improvement-lgbm-router-v2-2026-04-14.md`; baseline `lgbm_train_stars_v1` sigue siendo el mejor cold model disponible
- **Direction B pendiente re-run**: `lgbm_router_v6` tiene validacion excelente (band 0 MAE 1.1315, -0.0273 vs baseline) pero falta re-run completo para generar artefactos de submission; es el experimento de mejora cold mas prometedor ejecutado hasta ahora
- **Direction D2 en curso**: `known_user_deep_router_v7_mae_v2` — MAE loss + lr=3e-4; es la hipotesis mas solida para superar v2_eval_v3 (0.6694): la curva monotona de D1 confirma la tesis de alineacion de loss, solo falta ajustar el LR
- **Siguientes pasos si D2 mejora**: si deep_mae < 0.6694, combinar lgbm_router_v6 (cold) + v7_mae_v2 (deep) en una submission unificada
- **Siguientes pasos si D2 no mejora**: explorar scheduler de LR (cosine decay o OneCycleLR) para navegar el ruido inicial del MAE loss sin overshooting
- **Causa raiz de las limitaciones del deep model ya identificada**: `smooth_l1_loss` — cambiada a `l1_loss` en `model/known_user_deep_e2e.py`; este cambio es permanente y afecta a todas las familias de configuracion futuras

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
- router known-user `two tower`: [`train_known_user_two_tower_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_known_user_two_tower_router.py)

### Situacion Arquitectonica

- La arquitectura deep vigente sigue usando una `business_tower` compartida para historial y candidato, un encoder de eventos temporales y una correccion acotada sobre `incumbent_prediction_raw`.
- La linea `known_user_deep_e2e` ya no es un experimento desconectado: en `v2_eval_v2` y `v2_eval_v3` queda activada en todas las bandas de usuario conocido.
- La tanda `v3` fue el mejor salto practico hasta ahora porque reforzo el experto corto `2-5` sin partirlo; su mejor run fue `runA_2_5_gate_looser`.
- La tanda `v4` introdujo un split interno `2-3` / `4-5`, mejoro la observabilidad de corto historial y confirmo que `4-5` responde mejor que `2-3`, pero no mejoro el MAE global frente a `v3`.
- La linea `two tower + cross + prefix memory` ya fue ejecutada con GPU en `known_user_two_tower_router_v2_eval_v2`, pero no activo ninguna banda en validacion y no desplazo al incumbent.
- El codigo actual de `known_user_deep_e2e` ya incorpora:
  - cinco expertos internos por banda efectiva
  - una correccion acotada sobre `incumbent_prediction_raw`
  - trazas de `alpha` y magnitud de correccion por banda
  - metricas explicitas para `2`, `3`, `4`, `5`, `2-3` y `4-5`
  - fix de consistencia para que incumbent y rama deep reconstruyan `history_band` con la misma logica

### Lectura Operativa Del Experimento `two tower`

- mejor run: `run02_structured_stable`
- `best_val_mae = 0.6890`
- bandas activadas: ninguna
- `deep_served_rows` en submission final: `0`

Conclusiones ya asumidas:

- la arquitectura `two tower` actual reduce RMSE en varias bandas, pero empeora MAE en todas
- el problema no es de cobertura sino de calibracion del corrector residual
- el incumbent oficial sigue siendo mejor punto de partida para submission
- la linea con mejor senal hoy sigue siendo `known_user_deep_router_v2_eval_v3`

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
- snapshot historico de router known-user deep sin activacion final:
  - [`known_user_deep_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v1)
- snapshot intermedio MoE ya util para comparacion:
  - [`known_user_deep_router_moe_eval_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_moe_eval_v1)
- snapshot corregido que habilita todas las bandas conocidas:
  - [`known_user_deep_router_v2_eval_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v2)
- snapshot candidato estable actual del router known-user deep:
  - [`known_user_deep_router_v2_eval_v3`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v3)
- snapshot experimental `v4` con split interno `2-3` / `4-5`, util para diagnostico pero no promovido:
  - [`known_user_deep_router_v4_eval_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v4_eval_v1)
- snapshot experimental `two tower` evaluado y no recomendado para submission:
  - [`known_user_two_tower_router_v2_eval_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_two_tower_router_v2_eval_v2)
- diagnostico especifico de la banda corta `2-5`:
  - [`known_user_short_band_diagnostic_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_short_band_diagnostic_v1)
- snapshot experimental `feature-first` entrenado con GPU para corto historial, util como ablation tabular y no promovido:
  - [`lgbm_feature_first_short_router_v1_gpu`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu)
- snapshot candidato o de referencia para comparacion de cabeza de entrenamiento:
  - [`competition_embeddings_v3_iter04`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04)
- run oficial downstream disponible:
  - [`frozen_embedding_regressor_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_v1)
- baseline router previo, ahora referencia historica:
  - [`lgbm_raw_router_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_v1)

## Resultado Operativo Mas Reciente De `known_user_deep_e2e`

Comparativa de los dos mejores snapshots recientes:

- `v3` mejor run: [`known_user_deep_router_v2_eval_v3/validation_summary.json`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v3/validation_summary.json)
  - mejor run: `runA_2_5_gate_looser`
  - `final_overall_mae = 0.5999163`
  - `overall_delta = -0.0034977`
  - delta en `2-5 = -0.0137899`
- `v4` mejor run: [`known_user_deep_router_v4_eval_v1/validation_summary.json`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v4_eval_v1/validation_summary.json)
  - mejor run: `runB_short_split_capacity`
  - `final_overall_mae = 0.6008307`
  - `overall_delta = -0.0025833`
  - delta en `2-5 = -0.0067465`
  - subtramos cortos:
    - `2-3 = -0.0058715`
    - `4-5 = -0.0087281`

Lectura:

- `v4` mejora el diagnostico y confirma heterogeneidad interna en la banda corta
- `4-5` responde mejor que `2-3`
- pero el split interno no supera al mejor experto corto unico de `v3`
- por tanto `v3` sigue siendo el snapshot candidato estable para submission

## Resultado Operativo Mas Reciente De La Tanda `feature-first`

Artefactos:

- run principal:
  - [`lgbm_feature_first_short_router_v1_gpu`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu)
- comparativa corta contra `v3`:
  - [`short_history_vs_v3.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu/short_history_vs_v3.csv)
- informe diagnostico previo que motivaba la tanda:
  - [`known_user_short_band_diagnostic_v1/report.md`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_short_band_diagnostic_v1/report.md)

Resumen:

- objetivo: probar una linea `feature-first` antes de seguir complicando el deep
- runtime: entrenamiento lanzado con `uv` y LightGBM en GPU
- mejora global frente al baseline tabular `lgbm_hybrid_conservative_v1`:
  - `0.6265079 -> 0.6254230`
  - delta global `-0.0010849`
- politica final de router:
  - `0 -> cold_model`
  - `6-20 -> known_prefix_deep_model`
  - `2-5 -> transition_blend_model`
  - resto de usuarios conocidos -> `known_model`

Lectura corta:

- la tanda valida que el nuevo bloque de features tabulares no rompe el router y si mejora un poco el global
- `known_prefix` vuelve a ser util en `6-20`
- pero la parte critica no sale bien: `2-5` empeora frente al `known_model` del propio run y queda claramente peor que `known_user_deep_router_v2_eval_v3`
- por eso el experimento no se promueve a submission

Comparativa contra `known_user_deep_router_v2_eval_v3` en banda corta:

- `2 = +0.04049` MAE
- `3 = +0.03613` MAE
- `4 = +0.03395` MAE
- `5 = +0.03425` MAE
- `2-3 = +0.03884` MAE
- `4-5 = +0.03408` MAE

Decision operativa:

- mantener `known_user_deep_router_v2_eval_v3` como referencia estable
- conservar esta tanda como ablation tabular y no como candidato de export
- reutilizar las nuevas features en iteraciones futuras, pero no insistir con el `transition_blend` actual para `2-5`

## Proximos Pasos Reales

- mantener `known_user_deep_router_v2_eval_v3` como referencia estable mientras no haya una iteracion que la supere
- usar `v4` como base de diagnostico para decidir la siguiente iteracion sobre corto historial
- priorizar `2-3`, que sigue siendo el subtramo mas flojo dentro de la zona corta
- si se retoma la linea tabular, hacerlo con una politica mas conservadora:
  - mantener las nuevas features
  - conservar `known_prefix` para `6-20`
  - desactivar o redisenar el `transition_blend` en `2-5`
- si se retoma la linea deep, probar a injertar las nuevas señales tabulares en el candidato `v3` en vez de sustituirlo por un router mas debil en corto historial
- decidir si la siguiente iteracion debe:
  - reforzar mas `2-3` con un experto especifico y mas conservador
  - o volver a un experto unico `2-5` mas fuerte, siguiendo la filosofia de `v3`
- si el gap de leaderboard sigue siendo grande, abrir despues una linea separada sobre `history_band = 0`
- mantener los snapshots oficiales y candidatos sincronizados con `docs/experiments/registry.md`
