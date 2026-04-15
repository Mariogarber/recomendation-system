# New Architecture Experiments — Direction A & B — 2026-04-14

- Proposito: registrar los experimentos de nueva arquitectura lanzados en la sesion tarde del 2026-04-14 y continuados en 2026-04-15.
- Tipo documental: `experiment`
- Fecha de ejecucion: `2026-04-14` (sesion tarde) — `2026-04-15` (continuacion: Dir C y D)
- Estado: `in-progress` — Dir A y B cerradas; Dir D (`v7_mae_v2`) en ejecucion

---

## Motivacion

Tras el ciclo de meta-stacking (v1-v6, cerrado) y el ciclo de mejora del cold model (v2-v5, cerrado), el analisis del sistema concluyo:

1. **Cold model**: el baseline `lgbm_train_stars_v1` sigue siendo el mejor disponible (band 0 MAE 1.1588). La unica via de mejora no explorada es usar embeddings de contenido del negocio directamente en las features del cold model, dado que los embeddings `competition_embeddings_v3_iter03` ya existen y tienen 128 dims de contenido semantico.

2. **Deep model**: el `KnownUserDeepE2EModel` tiene una limitacion estructural: el alpha gate fuerza la prediccion a ser `alpha * (correction acotada) + lgbm_incumbent`, donde alpha converge a 0.72–0.82 y la correccion maxima esta acotada por `correction_scale * tanh`. El modelo puede modificar como mucho un ~28% de la prediccion, lo que implica que aunque el deep component aprenda bien, su impacto maximo esta limitado por el diseno. Ademas el early stopping se dispara en epoch 4 de 20 por oscilacion de val MAE durante el aprendizaje inicial.

Los dos nuevos experimentos atacan estos problemas directamente:

- **Direction B** (`lgbm_router_v6`): anadir embeddings de contenido de negocio (PCA-32 desde `business_content_features.npz`) como features del cold LGBM
- **Direction A** (`known_user_deep_router_v5_direct_v1`): eliminar la restriccion del alpha gate y entrenar un predictor directo donde la correccion es completa y sin acotacion

---

## Direction B — Cold Model Con Embeddings De Contenido (`lgbm_router_v6`)

### Hipotesis

Los embeddings de negocio de `competition_embeddings_v3_iter03` capturan similitudes semanticas de categorias, atributos y texto que no estan accesibles para el cold model actual. Un usuario cold puede no tener historial, pero si tenemos un embedding del negocio candidato, podemos capturar "negocios de este tipo generalmente reciben una cierta calificacion de usuarios sin historial". Unido a los arquetipos de usuario (que si tienen informacion de perfil), la combinacion puede ser predictiva.

Las 32 dimensiones PCA reducen la dimensionalidad para evitar sobreajuste en un modelo LGBM con regularizacion estandar.

### Cambios De Codigo

**`utils/lgbm_raw_router_features.py`**:
- Nuevo campo `business_embedding_table: pd.DataFrame | None` (opcional, default `None`) en `RouterRawFeatureSpec`
- Nuevo parametro `business_embedding_table: pd.DataFrame | None = None` en `fit_router_feature_spec()`
- Logica de merge en `build_router_feature_frame()`: si el campo existe, hace left-join sobre `item == business_id` y agrega las columnas `biz_emb_00` … `biz_emb_31` (float32, fill con 0.0 si NaN)
- `__setstate__` para compatibilidad backward: los specs guardados antes de este cambio no tienen el campo y `joblib.load` necesita poder deserializar objetos viejos sin error

**`pipelines/lgbm/train_router.py`**:
- Import de `from scipy import sparse` y `from sklearn.decomposition import PCA`
- Nueva funcion `_load_business_embeddings_pca(embedding_root, n_components=32)`: carga `business_repr/business_content_features.npz`, aplica PCA-32, devuelve DataFrame con `business_id` + `biz_emb_00..31`
- Flag `--use-business-embeddings` (store_true): activa la carga + merge de embeddings en ambas llamadas a `fit_router_feature_spec()`

### Configuracion Del Run

```bash
uv run pipelines/lgbm/train_router.py \
  --save-root artifacts/lgbm_router_v6 \
  --no-biz-train-stats \
  --use-business-embeddings
```

- `--no-biz-train-stats`: ablado como confirmo v5; no usar biz_train_stats que empeoran el modelo
- `--use-business-embeddings`: activa los 32 embeddings PCA del contenido del negocio
- Todos los demas parametros: por defecto (64 arquetipos, all bands, lr/hiperparametros identicos a baseline)

**Features nuevas (+32)**:
- `biz_emb_00` … `biz_emb_31`: componentes PCA de `business_content_features.npz` (contenido semantico: atributos, categorias, horas, texto descriptivo)

### Estado

- `completed` — run activo (terminal `8e0391cf`) — completado en sesion 2026-04-14 (validacion solamente; submission incompleta por truncado del output con `Select-Object -First 30`)

### Resultados

- Cold band 0 MAE: **1.1315** (delta **-0.0273** vs baseline 1.1588) — mejor resultado cold hasta la fecha
- El run no genero artefactos de submission porque el proceso fue matado antes de la fase de submission
- Necesita re-run completo para generar `cold_submission_model.txt` y `submission_router_spec.joblib`

### Re-run Pendiente

```bash
uv run pipelines/lgbm/train_router.py \
  --save-root artifacts/lgbm_router_v6 \
  --no-biz-train-stats \
  --use-business-embeddings
```

---

## Direction A — Direct MAE Predictor (`known_user_deep_router_v5_direct_v1`)

### Hipotesis

La restriccion `pred = sigmoid(alpha) * correction_scale * tanh(correction) + lgbm_incumbent` limita artificialmente cuanto puede aprender el deep model. Los valores observados muestran alpha ~ 0.72-0.82 y correction_scale = 0.7–1.0, lo que significa que la correccion efectiva maxima es ~0.8 estrellas (cuando `correction_scale=1.0, alpha≈1, tanh≈1`). Para usuarios con patron de rating muy diferente al incumbent LGBM, esta acotacion impide que el deep model los corrija suficientemente.

En el modo directo: `pred = clamp(lgbm_incumbent + correction_logits, 1.0, 5.0)`. El `correction_logits` puede ser cualquier valor real, sin tanh ni alpha gate. El incumbent LGBM se convierte en un feature de entrada (sigue fluyendo por la arquitectura como antes) pero no como ancla estructural.

Ademas, se corrigen los problemas del regimen de entrenamiento:
- `lr = 1e-4` en lugar de `8e-4` (menos oscilacion en val MAE)
- `patience = 10` en lugar de `4` (el modelo puede sobrevivir el periodo inicial ruidoso)
- `max_epochs = 50` en lugar de `20` (tiempo suficiente para que converja)
- `auxiliary_loss_weight = 0.0` → eliminadas las perdidas BCE de like/dislike que fragmentan el gradiente
- `band_distillation_weights = 0` en todas las bandas → eliminada la perdida de distilacion que penaliza alejarse del incumbent

### Filosofia

En el modo corrector clasico (v1-v4), el modelo tiene un sesgo estructural en la funcion de perdida: la perdida de distilacion lo penaliza por alejarse del incumbent. Esto crea un conflicto entre "aprender" (main_loss → minimizar MAE) y "mantenerse cerca del incumbent" (distill_loss). En el modo directo este conflicto desaparece: el unico objetivo es minimizar la perdida principal (smooth_l1 sobre el rating objetivo).

### Cambios De Codigo

**`model/known_user_deep_e2e.py`**:
- Campo `use_direct_predictor: bool = False` en `KnownUserDeepE2EConfig` y `KnownUserDeepE2EArchitecture`
- Funcion `build_known_user_deep_e2e_architecture()`: pasa `use_direct_predictor` a la arquitectura
- Metodo `forward()`: si `use_direct_predictor`:
  - `correction_hat = expert_outputs["correction_logits"]` (sin tanh ni correction_scale)
  - `predicted_rating = clamp(incumbent_prediction_raw + correction_hat, 1.0, 5.0)`
- Funcion `compute_known_user_deep_loss()`: si `use_direct_predictor`, `alpha_regularization_weight` se fuerza a 0.0

**`utils/known_user_deep_e2e.py`**:
- `use_direct_predictor: bool = False` en `KnownUserDeepTrainingConfig`
- `to_model_config()` pasa `use_direct_predictor` a `KnownUserDeepE2EConfig`

**`pipelines/deep/train_known_user_deep.py`**:
- Nueva familia de configuracion `v5_direct_predictor`:
  - `lr=1e-4`, `patience=10`, `max_epochs=50`, `batch_size=512`
  - `auxiliary_loss_weight=0.0`
  - `band_distillation_weights = todas 0.0`
  - `band_correction_scales=None` (irrelevante en modo directo)
  - `use_direct_predictor=True`
- `--config-family` choices extendido con `v5_direct_predictor`

### Configuracion Del Run

```bash
uv run pipelines/deep/train_known_user_deep.py \
  --save-root artifacts/known_user_deep_router_v5_direct_v1 \
  --config-family v5_direct_predictor \
  --max-runs 1
```

El incumbent sigue siendo el default (`--incumbent-root artifacts/lgbm_raw_router_prefix_deep_v1`).

### Estado

- `completed` — run completado en sesion tarde 2026-04-14 (re-lanzado despues de fix de `__setstate__`)

### Resultados (Direction A — Cerrada, Regresion)

| Metrica | Valor |
|---|---|
| best_epoch | 1 |
| best_val_mae | 0.7167 |
| enabled_bands | >20 solamente |
| deep_mae overall | 0.6810 |
| vs incumbent LGBM | +0.0011 (peor) |
| vs v2_eval_v3 | +0.0116 (peor) |

**Diagnostico**: el modelo aprende en train (loss: 0.473→0.436 en 11 epochs) pero la val MAE explota en epoch 2 (0.7364) y nunca se recupera. Best epoch=1 significa que el primer paso de gradiente ya introduce inestabilidad.

**Causa raiz**: sin el alpha gate, `correction_logits` crece sin acotacion. El optimizador usa `smooth_l1_loss` para reducir el error de entrenamiento asignando correcciones cada vez mas grandes a los ejemplos dificiles. En validacion esas correcciones grandes se generalizan mal y el MAE se dispara.

**Conclusion**: el alpha gate (`tanh × correction_scale`) no es un bottleneck sino un **estabilizador arquitectonico**. La restriccion del modo directo no es el alpha gate: es la funcion de perdida `smooth_l1` que incentiva correcciones grandes.

---

## Direction C — Regularizadores Adicionales (`known_user_deep_router_v6_regularized`)

### Motivacion

Despues de cerrar Direction A, se identificaron dos hipotesis alternativas para el problema de overfitting:

1. **C1 (direct + L2 fuerte)**: mantener el modo directo pero con `weight_decay=1e-3` (50x mas fuerte que baseline). AdamW decae los pesos grandes, lo que deberia limitar la magnitud de `correction_logits` indirectamente.
2. **C2 (gated + escalas mas amplias)**: mantener el alpha gate pero ampliar `correction_scale` de 0.7–1.0 (v3) a 1.2–1.5. La hipotesis era que v3 era demasiado conservador y el gate mas amplio podria aprender mas.

### Resultados

| Run | best_epoch | best_val_mae | deep_mae | vs v2_eval_v3 | enabled_bands |
|---|---|---|---|---|---|
| C1 `runC1_direct_l2` | **5** | 0.7177 | 0.6782 | +0.0088 | 1, >20 |
| C2 `runC2_gated_wider` | 1 | 0.6978 | 0.6750 | +0.0056 | 1, 2-5, 6-20, >20 |

**C1**: best_epoch = 5 (mejora sobre Direction A que era epoch 1) — L2 retrasa el overfitting. Pero el deep_mae sigue siendo peor que v3.

**C2**: best_epoch = 1 — ampliar las escalas empeora la situacion; con mas espacio para correcciones el modelo se saturo mas rapido.

**Diagnostico perfeccionado**: la raiz del problema no es la magnitud de la correccion sino el **mismatch entre funcion de perdida y metrica de evaluacion**. `smooth_l1_loss` con `beta=1.0` se comporta como MSE para errores < 1.0 (la mayoria de las predicciones). El optimizador aprende a reducir errores cuadraticos grandes, lo que no alinea con minimizar MAE.

---

## Direction D — Loss MAE Directo (`known_user_deep_router_v7_mae_v*`)

### Hipotesis

Cambiar `F.smooth_l1_loss` → `F.l1_loss` en `compute_known_user_deep_loss()`. Alinea la funcion de perdida con la metrica de evaluacion. Mantener el alpha gate y las correction_scales de v3 (arquitectura estable).

### Cambio De Codigo

**`model/known_user_deep_e2e.py`** — una linea principal:
```python
# antes
main_loss = F.smooth_l1_loss(rating_pred[mask], rating_target[mask])
baseline_loss = F.smooth_l1_loss(outputs["baseline_hat"][mask], rating_target[mask])
correction_loss = F.smooth_l1_loss(correction_pred[mask], correction_target[mask])
# ahora
main_loss = F.l1_loss(rating_pred[mask], rating_target[mask])
baseline_loss = F.l1_loss(outputs["baseline_hat"][mask], rating_target[mask])
correction_loss = F.l1_loss(correction_pred[mask], correction_target[mask])
```

### v7_mae_v1 — Primer Run (Dir D1, lr=8e-4)

```bash
uv run pipelines/deep/train_known_user_deep.py \
  --save-root artifacts/known_user_deep_router_v7_mae_v1 \
  --config-family v7_mae_loss \
  --max-runs 1
```

Config: misma arquitectura que v3 (`correction_scales {"1":0.7, "2-5":0.95, "6-20":1.0, ">20":0.95}`), lr=8e-4, patience=6, max_epochs=25.

**Curva de aprendizaje**:

| epoch | train_loss | val_mae |
|---|---|---|
| 1 | 0.9711 | 0.6865 |
| 2 | 0.9619 | 0.6839 |
| 3 | 0.9593 | 0.6841 |
| 4 | 0.9576 | 0.6822 |
| 5 | 0.9557 | 0.6798 |
| 6 | 0.9532 | **0.6785** (best) |
| 7 | 0.9507 | 0.6821 |
| … | … | oscila |
| 12 | 0.9388 | 0.6851 (early stop) |

**La hipotesis se confirma**: curva monotona durante 6 epochs (vs peak inmediato en epoch 1 en Dir A/C). Train loss es ahora MAE real (~0.97 en vez de smooth_l1 ~0.47).

**Resultados**:

| Metrica | v7_mae_v1 | v2_eval_v3 |
|---|---|---|
| best_epoch | 6 | 4 |
| deep_mae | 0.6724 | **0.6694** |
| band 1 delta | -0.0069 | -0.0035 ← v7 **mejor** |
| band 2-5 delta | -0.0067 | **-0.0138** ← v3 mejor |
| band 6-20 delta | -0.0097 | **-0.0155** ← v3 mejor |
| band >20 delta | -0.0078 | **-0.0101** ← v3 mejor |

**v7_mae_v1 mejora banda 1 pero regresa en 2-5 y 6-20**. Causa:
1. `lr=8e-4` estaba calibrado para smooth_l1 (valores ~0.47). Con MAE (valores ~0.97), los gradientes son mayores → steps efectivos demasiado grandes → overshooting despues de epoch 6.
2. `correction_scales` agregaron banda 2-5 como una sola escala (0.95) en vez de las escalas separadas de v3 (`2-3: 0.9, 4-5: 0.95`).

### v7_mae_v2 — Run Corregido (Dir D2, lr=3e-4) — EN CURSO

```bash
uv run pipelines/deep/train_known_user_deep.py \
  --save-root artifacts/known_user_deep_router_v7_mae_v2 \
  --config-family v7_mae_loss \
  --max-runs 1
```

Config: `lr=3e-4` (~2.7x menor), `patience=10`, `max_epochs=40`, `correction_scales={"1":0.7, "2-3":0.9, "4-5":0.95, "6-20":1.0, ">20":0.95}` (exactamente como v3).

Estado: **en ejecucion** (sesion 2026-04-15).

---

## Resultados Esperados Y Criterio De Exito

| Experimento | Metrica objetivo | Estado | Resultado |
|---|---|---|---|
| `lgbm_router_v6` | Cold band 0 MAE < 1.1588 | Validacion OK | **1.1315 (-0.0273)** — necesita re-run para submission artifacts |
| `known_user_deep_router_v5_direct_v1` | Val MAE known < 0.6694 | **Cerrado** | 0.6810 — regresion; alpha gate confirmado como estabilizador |
| `known_user_deep_router_v6_regularized` (C1/C2) | Val MAE < 0.6694 | **Cerrado** | 0.6782 / 0.6750 — ambas peores; smooth_l1 confirmada como causa raiz |
| `known_user_deep_router_v7_mae_v1` (D1) | Val MAE < 0.6694 | **Cerrado** | 0.6724 — curva monotona confirmada; lr demasiado alto |
| `known_user_deep_router_v7_mae_v2` (D2) | Val MAE < 0.6694 | **En curso** | pendiente |
| `known_user_deep_router_v8_fixed` (leakage fix attempt) | Val MAE < 0.6694 | **CERRADO — REGRESION CRITICA** | deep_mae 0.927 — catastrofico |

---

## Experimento Fallido — `known_user_deep_router_v8_fixed` (2026-04-15)

### Hipotesis

Aplicar la correccion de `user_average_stars` (train-only) directamente en `train_known_user_deep.py`, antes de llamar a `_predict_incumbent_router()`. Idea: si el incumbent produce predicciones mas honestas, el deep model aprende correcciones mas limpias.

### Resultado

| Banda | incumbent_mae | deep_mae | delta |
|---|---|---|---|
| 1    | **1.161** | 1.112 | -0.049 |
| 2-5  | **0.903** | 0.932 | +0.029 ✗ |
| 6-20 | **0.710** | 0.776 | +0.066 ✗ |
| >20  | **0.609** | 0.610 | +0.001 |
| global (deep_model_eval) | **0.918** | **0.927** | **+0.009 ✗** |

Comparado con v7_mae_v1 (incumbent_mae banda 1 = 0.680, deep_mae = 0.672): **regresion de ~+0.25 MAE global**.

### Causa Raiz — Error de Diseno

El incumbent `lgbm_raw_router_prefix_deep_v1` fue **entrenado** con `user_average_stars` leaky (del fichero `usuarios.csv`, que incluye reviews del periodo de test). Cuando `train_known_user_deep.py` sobrescribe `users_df["average_stars"]` con valores derivados solo del train split antes de llamar a `_predict_incumbent_router()`, el incumbent LGBM recibe features **fuera de distribucion**.

El resultado es que las predicciones del incumbent se degradan masivamente (band 1 incumbent MAE: 0.680 → 1.161). El deep model entonces intenta aprender correcciones sobre predicciones ya corruptas, y como las correcciones target son ahora mucho mas grandes y ruidosas, el modelo diverge.

**Analogia**: es como cambiar la escala de un eje de entrada de un modelo ya entrenado. El modelo aprendio que `average_stars=4.2` significa "usuario con sesgo positivo". Si ahora ese campo vale 3.76 (global mean) para el mismo usuario, el modelo interpreta que es un usuario neutro y predice de forma completamente diferente.

### Leccion — Orden Correcto Para El Fix

El fix de leakage `user_average_stars` DEBE aplicarse en este orden:

1. **Primero**: reentrenar el incumbent LGBM con `train_router.py --cf-model-path ...` (que ya aplica `build_train_user_stars` correctamente desde el inicio del ciclo). Producir un nuevo artifact de incumbent honest, e.g. `lgbm_router_v9_cold_signals`.
2. **Despues**: entrenar el deep model pasando ese nuevo incumbent honest como `--incumbent-root artifacts/lgbm_router_v9_cold_signals`.
3. Solo entonces el incumbent y el deep model comparten la misma distribucion de features.

El incumbent y el deep model son **parejas acopladas** — no se puede parchear uno sin reentrenar el otro.

### Accion Correctiva

- Revertido el override de `build_train_user_stars` en `train_known_user_deep.py` y `predict_known_user.py`.
- Añadido comentario explicativo en ambos ficheros.
- El fix de leakage en `train_router.py` (que ya estaba correcto) se mantiene.
- Proximo paso: lanzar `lgbm_router_v9_cold_signals` (con CF bias + friend features + honest stars), obtener un incumbent honesto, y entonces reentrenar el deep model contra ese nuevo incumbent.

---

## Cambios De Infraestructura Adicionales

- `RouterRawFeatureSpec.__setstate__`: compatibilidad backward completa para cualquier spec serializado antes de este PR; necesario porque el deep training carga el spec del incumbent desde disco
- `lgbm_raw_router_features.py` field ordering: `business_embedding_table` como campo opcional final con `field(default=None)` del modulo `dataclasses`
