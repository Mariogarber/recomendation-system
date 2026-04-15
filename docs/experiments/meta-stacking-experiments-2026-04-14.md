# Meta-Stacking Experiments 2026-04-14

- Proposito: registrar el ciclo completo de experimentos de meta-stacking (v1-v6) ejecutados sobre la rama content-based para intentar superar el techo de 0.6528 en leaderboard.
- Tipo documental: `experiment`
- Fecha de ejecucion: `2026-04-14`
- Estado: `closed` — ningun experimento supero al incumbent; linea abandonada

---

## Contexto

El mejor resultado conocido en este punto era `meta_lgbm_hybrid_v1` con LB 0.6528.
El meta-modelo toma predicciones del CB router como feature principal y entrena un LightGBM corrector que combina esa prediccion con features de CF (bias de usuario/negocio) y features del usuario.

El objetivo de la sesion fue: explorar si mas features, mas versiones del meta-modelo o estrategias mas sofisticadas de combinacion podian superar ese techo.

Se detecto durante el analisis que el 41% de las filas de test corresponde a usuarios cold (band `0`), y que ningun meta-modelo de esta familia podia corregirlos directamente porque el corrector solo tiene senales debiles para cold users.

---

## Arquitectura Comun

```
train_reviews -> CF bias (user_bias, item_bias) -> meta_features
             -> CB router prediction (known_user_deep_router_v2_eval_v3) -> cb_pred
             -> LightGBM corrector -> final_prediction
```

El CB router ya contiene internamente tres ramas:
- `known_model`: LightGBM para usuarios con historial alto (>=6 reviews)
- `cold_model`: LightGBM con arquetipos para usuarios band `0`
- `known_prefix_model`: LightGBM con features deep-prefix para bandas 1–2–5

El meta-modelo actua solo como una capa de correccion sobre la salida del router. No puede sustituir la logica cold interna.

---

## Version v1 — Baseline Meta (CB + CF)

- artefacto: [`meta_lgbm_hybrid_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v1)
- script: `pipelines/collaborative_filtering/train_meta.py --version 1`
- features: `cb_pred`, `user_bias`, `item_bias` (3 features)
- entrenamiento: LightGBM MAE sobre `train_split`, val sobre `val_split`
- val MAE (global): ~0.665
- **LB: 0.6528** ← mejor resultado de la sesion; nunca superado
- interpretacion: la combinacion CF bias + CB pred aporta algo sobre CB solo, pero el 41% de cold users no recibe correccion util

---

## Version v2 — Direction 2 Fix + User/Item Bias Features

- artefacto: [`meta_lgbm_hybrid_v2`](/C:/Users/mario/OneDrive/Documentos/UPM\Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v2)
- hipotesis: el CF bias calculado con direction=1 tenia un error de signo; corregirlo deberia mejorar el meta
- resultado: val MAE 0.687 > baseline 0.669 — peor
- causa: el MAE del CF era 1.004, demasiado ruidoso para complementar al CB
- estado: `deprecated`

---

## Version v3 — CB solo + Bias Features Sin CF

- artefacto: [`meta_lgbm_hybrid_v3`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v3)
- hipotesis: quitar CF completamente, dejar solo CB pred + features de usuario/negocio
- features: `cb_pred`, `user_average_stars`, `business_stars`, `user_review_count_log1p`, etc.
- resultado: val MAE 0.673 vs CB rounded 0.669 — delta +0.004, no hay mejora
- interpretacion: sin la senal CF de bias no hay suficiente informacion para corregir
- estado: `deprecated`

---

## Version v4 — 19 Features Expandidas

- artefacto: [`meta_lgbm_hybrid_v4`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v4)
- hipotesis: anadir mas features (engagement, temporales, fraccion de prediccion CB) mejora el corrector
- features nuevas sobre v1: `user_total_votes_log1p`, `cb_pred_fractional_part`, `correction_hat`, `useful_norm`, `funny_norm`, `review_month`, `review_weekday`, 5 features mas
- val MAE: 0.6640 (delta -0.0054 vs CB rounded 0.6695 — mejora local real)
- **LB: 0.6529** — sin mejora respecto a v1
- importancias destacadas:
  - `cb_pred_fractional_part`: ganancia 201k (mayor contribuidor)
  - `correction_hat`: ganancia 112k
  - `user_total_votes_log1p`: ganancia 40k
  - features engagement/temporales: ganancia casi cero
- diagnostico: la mejora local existe pero no se traslada porque las cold rows (41%) no mejoran con el corrector
- estado: `candidate` (mejor meta local, pero no supera LB)

---

## Version v5 — Joint All-Bands Meta

- artefacto: [`meta_lgbm_hybrid_v5`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v5)
- hipotesis: entrenar un unico meta-modelo para todas las bandas (known + cold juntas) con features especificas de cold
- features cold anadidas: prior bayesiano de negocio, mean/std de estrellas del negocio en train
- resultado: val MAE 0.914, **LB 0.8565** — catastrofico
- causa raiz identificada: el modelo reemplazaba las predicciones cold del CB router (que usa `cold_submission_model.txt` entrenado con arquetipos) con un prior bayesiano debil. El 41% de test rows perdio la prediccion de calidad.
- lecciones:
  - el `cold_submission_model.txt` de `lgbm_train_stars_v1/` ya es el mejor predictor disponible para cold users
  - cualquier meta-modelo que sobreascriba esas predicciones con una senal mas debil escala el error masivamente
  - el prior bayesiano de negocio es una senal mucho mas debil que el modelo con arquetipos
- estado: `deprecated`

---

## Version v6 — Dos Modelos Dedicados (Known + Cold)

- artefacto: [`meta_lgbm_hybrid_v6`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/meta_lgbm_hybrid_v6)
- hipotesis: separar el meta en dos LightGBMs — uno para usuarios conocidos, otro para cold; evitar que el corrector cold sea debil
- modelo known: val MAE 0.665 — comparable a v1
- modelo cold: val MAE 1.017 — muy alto; confirma que el meta-corrector cold es ineficaz
- prediccion LB esperada: todavia mala por el mismo motivo que v5 (la rama cold del meta sigue siendo mas debil que `cold_submission_model.txt`)
- estado: `deprecated` — no se subio al leaderboard

---

## Conclusion Critica Del Ciclo

### Por Que El Meta-Stacking No Mejoro

El CB router (`known_user_deep_router_v2_eval_v3`) encapsula internamente tres modelos LightGBM:

```
cold rows  (band 0)   -> cold_submission_model.txt  (arquetipos + features de negocio)
short rows (band 1-5) -> known_prefix_submission_model.txt
long rows  (band >=6) -> known_submission_model.txt
```

El `cold_submission_model.txt` usa 64 arquetipos K-means sobre 9 features de perfil de usuario y 6 tablas de afinidad arquetipo-negocio. Es un modelo especializado y entrenado, no un prior simple.

El meta-modelo en v5/v6 intentaba sobrescribir esas predicciones con una senal mucho mas debil (prior bayesiano de negocio). Resultado: MAE cold sube de ~1.25 a ~2.0+, y como cold es 41% del test, el LB sube de 0.6528 a 0.8565.

### Techo Real Del Meta-Stacking Para Known Users

Sobre la parte known del test (59%), v4 logro una pequena mejora local (delta -0.0054), pero no fue suficiente para mover el LB. La razon probable es que el CB router ya es muy bueno para usuarios conocidos, y la senal residual que puede capturar el meta-corrector es minima.

### Pivot Decidido

En lugar de continuar con meta-stacking, el trabajo se pivoto a mejorar el cold model subyacente directamente en `train_router.py`. Eso es lo que registra el documento `cold-model-improvement-lgbm-router-v2-2026-04-14.md`.

---

## Resumen De Resultados

| Version | Val MAE | LB MAE | Estado |
|---|---:|---:|---|
| v1 (CB+CF) | ~0.665 | **0.6528** | `candidate` — mejor LB |
| v2 (fix direction) | 0.687 | — | `deprecated` |
| v3 (CB+bias, sin CF) | 0.673 | — | `deprecated` |
| v4 (19 features) | 0.6640 | 0.6529 | `candidate` — mejor val |
| v5 (joint all-bands) | 0.914 | 0.8565 | `deprecated` — catastrofico |
| v6 (dos modelos) | 0.665/1.017 | — | `deprecated` |
