# Two-Tower Router 2026-04-12

- Proposito: registrar la primera evaluacion completa de la arquitectura `known_user_two_tower_cross`.
- Tipo documental: `experiment`
- Fecha de ejecucion: `2026-04-12`

## Setup

- script: [`train_known_user_two_tower_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_known_user_two_tower_router.py)
- dispositivo: `CUDA`
- GPU detectada: `NVIDIA GeForce RTX 3060 Laptop GPU`
- incumbent de referencia: [`lgbm_raw_router_prefix_deep_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_raw_router_prefix_deep_v1)
- artefacto exportado: [`known_user_two_tower_router_v2_eval_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_two_tower_router_v2_eval_v2)

## Resultado Corto

- mejor run: `run02_structured_stable`
- `best_val_mae = 0.6890317798`
- `best_val_rmse = 1.0889508724`
- bandas activadas: ninguna
- `deep_served_rows` en submission final: `0`

## Comparacion Por Bandas

La mejor run empeoro el MAE en todas las bandas conocidas:

| Banda | Incumbent MAE | Two-Tower MAE | Delta |
|---|---:|---:|---:|
| `1` | `0.6802` | `0.6860` | `+0.0058` |
| `2-5` | `0.7161` | `0.7242` | `+0.0081` |
| `6-20` | `0.6601` | `0.6707` | `+0.0106` |
| `>20` | `0.5835` | `0.6046` | `+0.0212` |

## Interpretacion

Hallazgos principales:

- la arquitectura tuvo cobertura total en usuarios conocidos, asi que el problema no fue de disponibilidad
- el corrector residual fue demasiado agresivo para el objetivo de MAE
- el modelo parece mover predicciones hacia menor RMSE en algunas bandas, pero no hacia mejor error absoluto medio
- la peor degradacion aparece en `>20`, lo que sugiere que el bloque `cross` actual no esta calibrando bien ni siquiera donde hay mas historia

## Incidencias Durante La Ejecucion

Se detectaron y corrigieron dos fallos en el script:

- referencia a `router_branch` donde el frame contenia `incumbent_branch`
- referencia a `prediction` donde la salida incumbent exponia `incumbent_prediction_raw`

Ambos fallos eran de ensamblado del experimento, no de definicion matematica del modelo.

## Conclusiones

- la familia `two tower` queda registrada como evaluada, pero no promovida
- el router oficial no debe cambiar por este resultado
- la linea que sigue mereciendo iteracion es `known_user_deep_router_v2_eval_v2`

## Siguiente Intento Recomendado

- reducir la amplitud maxima de correccion por banda
- penalizar mas los casos donde la rama deep empeora al incumbent
- bajar el `alpha` por defecto en `2-5`, `6-20` y `>20`
- considerar una version donde el `two tower` actue como reranker o calibrador de baja amplitud en lugar de sustituto parcial
