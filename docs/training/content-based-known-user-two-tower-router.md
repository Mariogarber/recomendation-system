# Two-Tower Router De Content-Based

- Proposito: documentar el entrenamiento y la logica operativa de `train_known_user_two_tower_router.py`.
- Tipo documental: `reference`
- Ultima actualizacion: `2026-04-12`

## Resumen

`content-based/train_known_user_two_tower_router.py` no sustituye el router oficial.

Lo que hace es:

1. cargar el incumbent `lgbm_raw_router_prefix_deep_v1`
2. reconstruir su prediccion para train, validacion y test
3. entrenar una rama `known_user_two_tower_cross` solo para usuarios conocidos
4. comparar esa rama con el incumbent por bandas de historial
5. activar la rama nueva solo si mejora MAE por banda
6. reentrenar la mejor configuracion y exportar una submission mixta

La politica de fallback es estricta:

- si una banda no mejora, no se activa
- si una fila no tiene prediccion deep, se mantiene el incumbent

## Arquitectura Resumida

La rama `two tower` combina:

- una `business_tower` compartida para candidato e historial
- un `event_encoder` para cada interaccion historica
- un `user_context_encoder` con metadata numerica, auxiliar y categorica
- varias memorias de prefijo
- un bloque `cross network`
- una salida residual sobre `incumbent_prediction_raw`

La prediccion final sigue la idea:

- `pred = clip(incumbent_prediction_raw + alpha * correction_hat, 1, 5)`

## Configuraciones Auditadas

El script prueba hasta tres runs:

- `run01_structured_base`
- `run02_structured_stable`
- `run03_structured_capacity`

En la evaluacion auditada de `2026-04-12`, la mejor fue:

- `run02_structured_stable`
- `best_epoch = 3`
- `best_val_mae = 0.6890`
- `best_val_rmse = 1.0890`

## Resultado Del Experimento Auditado

Artefacto generado:

- [`known_user_two_tower_router_v2_eval_v2`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_two_tower_router_v2_eval_v2)

Resumen operativo:

- bandas activadas: ninguna
- `deep_served_rows` en submission final: `0`
- la submission resultante no mejora operativamente al incumbent

Comparacion por banda frente al incumbent en la mejor run:

- `1`: `delta_mae = +0.0058`
- `2-5`: `delta_mae = +0.0081`
- `6-20`: `delta_mae = +0.0106`
- `>20`: `delta_mae = +0.0212`

Lectura:

- el modelo no fallo por falta de cobertura
- fallo por calibracion y sobrecorreccion
- aunque redujo RMSE en varias bandas, el criterio de seleccion es MAE y ahi perdio en todas

## Incidencias Corregidas Durante La Ejecucion

Durante la corrida auditada se corrigieron dos errores del script:

- uso de `router_branch` en un frame que exponia `incumbent_branch`
- uso de una columna `prediction` inexistente al reconstruir la submission final

Tras esos fixes, el entrenamiento completo con GPU pudo terminar y exportar artefactos.

## Conclusiones

- esta primera iteracion `two tower` no debe promoverse a snapshot recomendado
- el incumbent oficial sigue siendo la mejor base de submission
- la mejor linea experimental para usuarios conocidos sigue siendo `known_user_deep_router_v2_eval_v2`

Si se reintenta esta familia, la siguiente iteracion deberia:

- imponer una correccion residual aun mas pequena
- regularizar mas el `alpha`
- optimizar de forma mas directa contra MAE
- comparar contra el `known_user_deep_e2e` fuerte, no solo contra el incumbent oficial
