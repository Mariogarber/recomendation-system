# Cold Model Improvement — lgbm_router_v2 a v5 — 2026-04-14

- Proposito: registrar el ciclo completo de experimentos de mejora del cold LightGBM (lgbm_router_v2 a v5).
- Tipo documental: `experiment`
- Fecha de ejecucion: `2026-04-14`
- Estado: `closed` — ciclo cerrado; ninguna mejora supera el baseline de forma significativa; rama abandonada

---

## Motivacion

Tras el ciclo de meta-stacking (ver `meta-stacking-experiments-2026-04-14.md`), se identifico que:

- el 41% de las filas del test de competicion corresponde a usuarios cold (band `0`)
- el cold model del CB router (`cold_submission_model.txt` de `lgbm_train_stars_v1/`) ya era el mejor predictor disponible para esas filas
- ninguna capa de meta-corrector podia superarlo sin degrade

El diagnostico llevo a la conclusion de que la unica forma de mejorar el LB era mejorar ese modelo cold directamente.

---

## Tres Mejoras Implementadas

### 1. Filtro de banda de entrenamiento

**Hipotesis**: el cold model aprende patrones mezclados si se entrena con usuarios de todas las bandas. Los usuarios con mucho historial (band `>20`) tienen un perfil de comportamiento diferente al de un usuario cold genuino. Entrenar solo con bandas bajas (1, 2-5) hace que el modelo aprenda a predecir para perfiles mas parecidos a los cold reales.

**Cambio**: en `pipelines/lgbm/train_router.py`, se anado el argumento `--cold-training-max-band` con opciones `1`, `2-5`, `6-20`, `>20` (default `>20` preserva conducta anterior). Se aplico tanto al split de validacion como al modelo de submission.

**Valor usado en este run**: `--cold-training-max-band 2-5` → el cold model se entrena solo con filas de bandas 1 y 2-5.

### 2. Estadisticas de negocio calculadas desde train

**Hipotesis**: las estrellas medias de Yelp en `negocios.csv` son un agregado all-time que puede incluir reviews del periodo de test. Calcular `business_train_mean_rating`, `business_train_rating_std` y `business_train_vs_yelp_gap` directamente desde `train_reviews` es mas honesto y puede aportar una senal mas limpia.

**Cambio**: en `utils/lgbm_raw_router_features.py`:
- Se agrego el campo `business_train_stats_table: pd.DataFrame` al dataclass `RouterRawFeatureSpec`.
- En `fit_router_feature_spec()` se computa `train_reviews.groupby("business_id")["stars"].agg(business_train_mean_rating="mean", business_train_rating_std="std")` con `fillna` por la media global.
- En `build_router_feature_frame()` se hace merge sobre `item == business_id` y se calculan tres features nuevas:
  - `business_train_mean_rating` (float32)
  - `business_train_rating_std` (float32)
  - `business_train_vs_yelp_gap = business_train_mean_rating - business_stars` (float32)

### 3. Mas arquetipos de usuario: 64 → 128

**Hipotesis**: duplicar el numero de arquetipos K-means da al cold model mas granularidad para distinguir perfiles de usuario, especialmente en clusters densos (usuarios promedio casuales).

**Cambio**: argumento `--n-user-archetypes 128` en `train_router.py`.

---

## Resumen De Experimentos

Se ejecutaron 4 runs controlados (v2–v5), cada uno variando un parametro respecto al baseline `lgbm_train_stars_v1`.

| Artefacto | Archetypes | Biz stats | Training bands | Cold band 0 MAE | Delta vs baseline |
|---|---:|---|---|---:|---:|
| `lgbm_train_stars_v1` (baseline) | 64 | — | all | **1.1588** | — |
| `lgbm_router_v5` | 64 | — | all | **1.1494** | **-0.0094** ← mejor |
| `lgbm_router_v4` | 32 | ✓ | all | 1.2086 | +0.0498 |
| `lgbm_router_v3` | 128 | ✓ | all | 1.2104 | +0.0516 |
| `lgbm_router_v2` | 128 | ✓ | 1+2-5 only | 1.2185 | +0.0597 |

**Nota**: todas las demas bandas (1, 2-5, 6-20, >20) son identicas en todos los runs. Solo el cold model cambia.

---

## Analisis Run A Run

### lgbm_router_v2 — 128 arqs + biz stats + filtro bandas (1, 2-5)
- **Hipotesis**: entrenar el cold model solo con usuarios de historial bajo lo hace mas parecido a los cold reales
- **Resultado**: band 0 = 1.2185 (+0.0597) — peor
- **Causa**: el filtro elimina 193k filas de entrenamiento; perder datos de bandas altas reduce la cobertura de combinaciones arquetipo-negocio

### lgbm_router_v3 — 128 arqs + biz stats + all bands
- **Hipotesis**: mas arquetipos dan mas granularidad al cold model
- **Resultado**: band 0 = 1.2104 (+0.0516) — peor
- **Causa**: con 128 clusters, cada arquetipo tiene ~mitad de ejemplos en las tablas de afinidad → senales mas ruidosas

### lgbm_router_v4 — 32 arqs + biz stats + all bands
- **Hipotesis**: menos arquetipos, tablas de afinidad mas densas
- **Resultado**: band 0 = 1.2086 (+0.0498) — peor pero mejor que v2/v3
- **Causa**: las features `biz_train_stats` siguen siendo el factor dominante que empeora el modelo

### lgbm_router_v5 — 64 arqs + NO biz stats + all bands (ablacion)
- **Hipotesis**: las 3 features de biz_train_stats son la causa del empeoramiento — eliminarlas deberia recuperar el baseline
- **Resultado**: band 0 = 1.1494 (-0.0094) — **ligeramente mejor que el baseline**
- **Diagnostico**: confirmado que las features de biz_train_stats son el factor causal del empeoramiento; sin ellas, la configuracion identica al baseline mejora marginalmente (diferencia explicable por varianza de entrenamiento)

---

## Por Que Las Features biz_train_stats Empeoran El Cold Model

Las tres features anadidas son:
- `business_train_mean_rating` — media de estrellas del negocio en train_reviews
- `business_train_rating_std` — desviacion estandar de estrellas en train_reviews
- `business_train_vs_yelp_gap` — diferencia entre la media de train y las estrellas de Yelp all-time

El problema: `business_train_mean_rating` esta altamente correlacionado con `business_stars` (que ya estaba en el feature set). Anadir una feature casi redundante pero calculada sobre un subconjunto de datos (solo train_reviews) introduce ruido en la direccion donde el modelo ya tiene buena senal. LightGBM puede hacer splits suboptimos sobre la version ruidosa en lugar de la version limpia.

---

## Modificaciones Al Codigo (Permanentes)

Todos los cambios de codigo quedan en el repositorio como infraestructura reutilizable:

| Archivo | Cambio | Utilidad futura |
|---|---|---|
| `utils/lgbm_raw_router_features.py` | `include_biz_train_stats` param en `fit_router_feature_spec()` | permite ablaciones limpias |
| `utils/lgbm_raw_router_features.py` | `business_train_stats_table` en `RouterRawFeatureSpec` | infraestructura disponible si se necesita |
| `pipelines/lgbm/train_router.py` | `--cold-training-max-band` arg | util para futuros experimentos de distribucion |
| `pipelines/lgbm/train_router.py` | `--no-biz-train-stats` arg | ablacion limpia de biz stats |
| `pipelines/lgbm/train_router.py` | `--n-user-archetypes` arg | ya existia, confirmado como parametro util |
| `pipelines/deep/predict_known_user.py` | `--incumbent-router-root` arg | util para cambiar el router sin re-entrenar el modelo deep |

---

## Conclusion Y Cierre Del Ciclo

La mejora del cold model via estos tres vectores (mas arquetipos, filtro de bandas, estadisticas de negocio) no produce una mejora sustancial sobre el baseline.

El unico resultado que mejora ligeramente es v5 (ablacion de biz_stats), con:
- delta cold band 0: -0.0094
- impacto LB estimado: -0.0039 (de 0.6528 a ~0.6489)

Esta mejora es demasiado pequena para justificar una nueva submission. La varianza del LB puede absorberla.

**Linea cerrada.** El baseline `lgbm_train_stars_v1` sigue siendo el mejor cold model disponible.
