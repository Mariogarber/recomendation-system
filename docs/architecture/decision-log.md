# Decision Log De Arquitectura

- Proposito: registrar decisiones arquitectonicas relevantes, su estado y su impacto sobre codigo, flows y artefactos.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-16`

## Como Leer Este Log

Estados usados:

- `proposed`
- `accepted`
- `implemented`
- `superseded`
- `rejected`

## ADR-001: `docs/` pasa a ser la casa canónica de documentacion

- Estado: `implemented`
- Fecha: `2026-04-10`
- Contexto:
  - la documentacion estaba fragmentada entre `README.md`, `content-based/README.md`, `docs/` y documentos largos de modulo
- Decision:
  - la documentacion transversal pasa a vivir en `docs/`
  - los `README.md` quedan como hubs cortos
- Consecuencias:
  - se reducen duplicaciones
  - los documentos antiguos pasan a ser stubs de transicion

## ADR-002: La arquitectura deep vigente mantiene una `business_tower` compartida

- Estado: `implemented`
- Fecha: `2026-04-10`
- Contexto:
  - la implementacion actual del deep encoder reutiliza la misma torre para historial y candidato
- Decision:
  - documentar explicitamente esa arquitectura como estado actual

## ADR-003: El frozen regressor se considera etapa oficial downstream del pipeline

- Estado: `implemented`
- Fecha: `2026-04-10`
- Contexto:
  - el script downstream existia pero no estaba tratado como etapa oficial en la narrativa principal
- Decision:
  - incorporar `train_frozen_embedding_regressor.py` y su modelo asociado a los flows y training del modulo

## ADR-004: Separar encoding de historial y encoding del negocio candidato

- Estado: `proposed`
- Fecha: `2026-04-10`
- Contexto:
  - hoy el `user_embedding` no recibe directamente el candidato, pero historial y candidato siguen compartiendo la misma `business_tower`
  - eso deja un acoplamiento indirecto entre ambos roles
- Decision propuesta:
  - introducir una torre para negocios del historial y otra para el negocio candidato
  - mantener `user_embedding` dependiente solo de historial agregado y metadata
- Alternativas consideradas:
  - mantener una sola torre y solo clarificar la separacion conceptual
  - mover el sistema a una arquitectura `interaction-first`
- Impacto esperado:
  - separar mejor preferencias del usuario y representacion del item a predecir
  - obligar a regenerar embeddings deep y checkpoints
- Seguimiento requerido si se implementa:
  - actualizar arquitectura, flows, training, contratos de artefactos y registro de experimentos

## ADR-005: Router prefix-deep para usuarios conocidos de historia corta y media

- Estado: `implemented`
- Fecha: `2026-04-11`
- Contexto:
  - el router `raw_core` + arquetipos mejoro la captura de cold start, pero seguia habiendo margen en usuarios conocidos intermedios
  - los intentos deep anteriores no daban una mejora estable como scorer global, pero si dejaban un bundle deep utilizable como fuente de embeddings
  - el run nuevo mostro que el candidato prefix-deep solo merece activarse en `6-20` con un margen minimo de mejora
- Decision:
  - mantener `cold_model` para `history_band = 0`
  - mantener `known_model` como fallback para usuarios conocidos que no entren en la activacion prefix-deep
  - activar `known_prefix_deep_model` solo para `history_band = 6-20`
  - considerar `lgbm_raw_router_prefix_deep_v1` como snapshot oficial del router
- Evidencia:
  - `validation_mae_rounded = 0.6265079379`
  - baseline previo `lgbm_raw_router_v1 = 0.6268747449`
  - `6-20` mejora a `0.6845791340`
  - `1` empeora a `0.7012391090`
  - `2-5` mejora a `0.7327732444` pero no supera el margen de activacion
- Consecuencias:
  - el router actual queda como combinacion de `raw_core`, arquetipos y prefix-deep
  - la banda `2-5` sigue siendo la mejor candidata para la siguiente iteracion de perfilado de usuario
  - los docs de estado, training y artefactos deben tratar `lgbm_raw_router_v1` como referencia historica, no como snapshot oficial vigente

## ADR-006: Meta-stacking sobre CB router abandonado

- Estado: `implemented`
- Fecha: `2026-04-14`
- Contexto:
  - se exploraron 6 versiones del meta-modelo LightGBM que tomaba la prediccion del CB router como feature principal
  - v1 con 3 features (cb_pred, user_bias, item_bias) produjo el mejor LB conocido: 0.6528
  - v4 con 19 features mejoro el val MAE local pero no el LB
  - v5 intentando corregir cold users con prior bayesiano produjo LB 0.8565 (catastrofico)
- Decision:
  - abandonar la linea meta-stacking sobre el CB router
  - el `cold_submission_model.txt` del router es mejor predictor cold que cualquier meta-corrector
  - el margen de mejora para known users via meta es < 0.001 en LB
- Consecuencias:
  - `meta_lgbm_hybrid_v1` queda como mejor submission LB conocida (0.6528) pero la arquitectura no se desarrolla mas
  - el foco se mueve a mejorar el incumbent LGBM directamente

## ADR-007: El gate sigmoidal es un estabilizador arquitectonico obligatorio

- Estado: `implemented`
- Fecha: `2026-04-14`
- Contexto:
  - se evaluo un modo directo (sin gate) en `known_user_deep_router_v5_direct_v1` y `v6_regularized`
  - todos los runs sin gate hicieron overfitting en epoch 1 (best_epoch=1)
  - L2 fuerte (weight_decay=1e-3) retrasó el overfitting a epoch 5 pero no lo elimino
- Decision:
  - mantener `pred = sigmoid(alpha) × correction_scale × tanh(correction) + incumbent` como formula canonica
  - el gate no es un bottleneck de capacidad: es el mecanismo que impide que `correction_logits` explote durante el entrenamiento
- Consecuencias:
  - cualquier nueva variante arquitectonica del corrector DEBE mantener el gate
  - la `correction_scale` puede variar por banda (tipico: 0.7–1.0) pero no eliminarse

## ADR-008: smooth_l1_loss → l1_loss para alinear loss con metrica

- Estado: `implemented`
- Fecha: `2026-04-14`
- Contexto:
  - `smooth_l1_loss` con beta=1.0 se comporta como MSE para errores < 1.0, que es la mayoria de las predicciones
  - el optimizador con smooth_l1 aprendia a reducir errores cuadraticos grandes, no MAE
  - cambio a `l1_loss` produjo curvas de aprendizaje monotones (6 epochs consecutivos sin subida) vs las oscilaciones anteriores
- Decision:
  - usar `F.l1_loss` como funcion de perdida principal en todas las nuevas familias de config del deep corrector
  - mantener `smooth_l1_loss` solo en experimentos que requieran backward-compatibility con checkpoints antiguos
- Consecuencias:
  - v7_mae y todas las familias lightweight/ultralight usan l1_loss
  - la estabilidad de la curva mejora pero el MAE final no supera v2_eval_v3 (el techo es de señal, no de loss)

## ADR-009: lr=2e-4 + batch=2048 como regimen estandar para corrector lightweight

- Estado: `implemented`
- Fecha: `2026-04-15`
- Contexto:
  - `lr=1e-3` con `correction_scale=1.0` producía oscilacion ±0.05 en val_mae en v_lightweight runA
  - la causa es la superficie no convexa de `sigmoid(alpha) × tanh(correction)`: gradientes grandes hacen overshooting
  - v_ultralight runB con `lr=2e-4 + batch=2048` produjo curva monotona 26 epochs consecutivos
- Decision:
  - para el corrector lightweight (embedding_dim=32), usar lr=2e-4 y batch_size=2048 como defaults
  - la `correction_scale` puede mantenerse en 1.0 para banda 6-20 (la estabilidad viene del lr, no del recorte de escala)
- Consecuencias:
  - runC (`runC_lw_emb32_stable_lr`) usa estos valores
  - todos los experimentos de familia lightweight futura deben empezar desde lr=2e-4

## ADR-010: user_average_stars de usuarios.csv no es leaky en esta competicion

- Estado: `implemented`
- Fecha: `2026-04-16`
- Contexto:
  - el split train/test de la competicion no es temporal estricto; ambos ficheros comparten el mismo rango de fechas
  - el experimento `v8_fixed` intento reemplazar user_average_stars por la media de train_reviews en un modelo ya entrenado → MAE banda 1 pasó de 0.680 a 1.161
  - analisis confirma que la feature no codifica el target ni es informativamente imposible de tener en produccion
- Decision:
  - usar `user_average_stars` de `usuarios.csv` directamente en todos los modelos, sin reemplazar por `build_train_user_stars`
  - `build_train_user_stars` queda disponible como utilidad experimental pero no se usa en el pipeline principal
- Consecuencias:
  - el pipeline de entrenamiento e inferencia no requiere ningun calculo adicional para esta feature
  - ver documentacion completa: [`reference/user-average-stars-leakage-analysis.md`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/user-average-stars-leakage-analysis.md)
