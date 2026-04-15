# Lightweight Deep Corrector — Experimento v_lightweight — 2026-04-15

- Proposito: evaluar si reducir drasticamente el numero de parametros del deep corrector (3.28M → ~200k) mejora la generalizacion en bandas de historial escaso.
- Tipo documental: `experiment`
- Fecha de ejecucion: `2026-04-15`
- Estado: `closed` — runA y runB completados; analisis disponible
- Artefacto: [`known_user_deep_lightweight_v1`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_lightweight_v1)

---

## Motivacion

El analisis del ciclo `v2_eval_v3` (mejor experimento deep hasta la fecha, val MAE 0.5999) revelo que todos los experimentos deep — independientemente de la arquitectura — convergen al mismo ceiling de mejora: **−0.002 a −0.004 delta sobre el incumbent LGBM** en overall MAE.

El conteo de parametros del modelo completo (`KnownUserDeepE2EModel`) es de **3.28M parametros** distribuidos de la siguiente manera:

| Modulo | Params | % total |
|---|---|---|
| business_tower (512→384→256) | 345k | 10.5% |
| 5× band residual hidden | 1,870k | 57.0% |
| 5× taste fusion | 775k | 23.6% |
| 3× attention heads | 198k | 6.0% |
| Otros (baseline, gates, heads) | 97k | 3.0% |

Con **337,862 ejemplos de entrenamiento**, el ratio es de **0.10 ejemplos/parametro** — territorio de underfitting severo para las bandas sparse. La banda `>20` tiene solo 6,021 filas de entrenamiento pero ocupa 280k params en sus expertos.

La hipotesis de este experimento es que reducir la capacidad del modelo puede mejorar la generalizacion en bandas con pocos datos, y que el task ceiling observado tiene origen en el ratio datos/parametros, no solo en la dificultad intriseca del task.

---

## Arquitectura Lightweight

Ver documentacion completa: [`docs/training/content-based-known-user-lightweight-deep.md`](/C:/Users/mario/OneDrive/Documentos/UPM\Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-known-user-lightweight-deep.md)

Cambios clave respecto al modelo completo:

- `embedding_dim`: 128 → **32**
- `business_hidden_layers`: (512, 384, 256) → **(64,)**
- `scorer_hidden_layers`: (256, 128) → **(64, 32)**
- `num_attention_heads`: 4 → **2**
- Mismo numero de bandas de expertos (5), misma estructura gated corrector
- Resultado: **~200k parametros** (1.7 ejemplos/param — ratio saludable)

---

## Configuracion De Runs

### runA — base (`dropout=0.20`, `weight_decay=1e-4`, con distilacion)

```
embedding_dim=32, event_hidden_dim=32, user_type_hidden_dim=32
scorer_hidden_dim=64, business_hidden_layers=(64,), scorer_hidden_layers=(64, 32)
num_attention_heads=2, dropout=0.20, batch_size=1024
learning_rate=1e-3, weight_decay=1e-4, max_epochs=40, patience=8
auxiliary_loss_weight=0.15
band_correction_scales: {"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95}
band_distillation_weights: {"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04}
```

### runB — alta regularizacion, sin distilacion (`dropout=0.25`, `weight_decay=5e-4`)

```
embedding_dim=32, dropout=0.25, batch_size=1024
learning_rate=1e-3, weight_decay=5e-4, max_epochs=40, patience=8
auxiliary_loss_weight=0.10
band_distillation_weights: None
```

---

## Resultados

### Training outcome

| Run | Best epoch | Best val MAE | Enabled bands |
|---|---|---|---|
| runA | 12 | 0.6717 | 1, **6-20** |
| runB | 5 | 0.6778 | 1 only |

runB paró en epoch 5 — la alta regularizacion impidio aprender los residuos de banda 6-20.

### Deep model evaluation (sobre val known, 64,727 filas)

| Run | Incumbent MAE | Deep MAE | Delta |
|---|---|---|---|
| runA | 0.6796 | 0.6780 | **−0.00156** |
| runB | 0.6796 | 0.6793 | −0.00034 |

### Por banda — runA (mejor run)

| Banda | Filas val | Incumbent MAE | Deep MAE | Delta |
|---|---|---|---|---|
| 1 | 21,144 | 0.6802 | 0.6802 | 0.000 |
| 2-5 | 23,568 | 0.7156 | 0.7155 | −0.000042 |
| **6-20** | **13,994** | **0.6595** | **0.6524** | **−0.00715** |
| >20 | 6,021 | 0.5835 | 0.5835 | 0.000 |

---

## Analisis

### 1. La hipotesis de parametros se confirma PARCIALMENTE en banda 6-20

La banda 6-20 obtiene **−0.00715 de mejora**, que es **superior al tipico −0.003 del modelo completo** en la misma banda. Con ~200k params y 13,994 ejemplos el ratio mejora a 0.07 ejemplos/param — aun sparse, pero la menor capacidad del modelo fuerza una representacion mas robusta.

### 2. Las bandas 1, 2-5 y >20 siguen siendo incorregibles

- **Banda 1** (21,144 filas): delta 0.0. Con un solo review, `user_average_stars` del perfil Yelp es el predictor optimo disponible. El corrector no tiene residuos aprendibles porque el incumbent LGBM ya usa esa feature directamente.
- **Banda 2-5** (23,568 filas): delta −4×10⁻⁵, esencialmente cero. Usuarios con historial corto son intrisecamente ruidosos; el incumbent ya extrae toda la señal disponible de sus features.
- **Banda >20** (6,021 filas): delta 0.0. Usuarios densos tienen MAE 0.5835, cerca del floor del task; no hay residuo sistematico que un corrector pueda aprender.

### 3. RunB sobreregularizó

`dropout=0.25 + weight_decay=5e-4 + sin distilacion` impidio que el modelo llegara a un estado util. La curva de val MAE no se aplanó hasta epoch 5, donde el modelo ya no tenia capacidad para mejorar sobre el incumbent en ninguna banda salvo la 1 (donde la mejora es cero de todas formas).

### 4. Comparacion con el modelo completo (v2_eval_v3/runA)

| Modelo | Params | Delta overall | Delta banda 6-20 | Bandas activadas |
|---|---|---|---|---|
| Full (v2_eval_v3/runA) | 3.28M | ~−0.0035 | ~−0.003 | 1, 2-5, 6-20, >20 |
| **Lightweight runA** | ~200k | **−0.00156** | **−0.00715** | 1, 6-20 |

El modelo lightweight consigue **mayor mejora en su banda fuerte (6-20)** pero activa menos bandas, resultando en menor delta overall. El modelo completo distribuye la mejora entre mas bandas pero con menor efecto por banda.

### 5. Diagnostico de task ceiling

El ceiling de mejora overall (~−0.002 a −0.004) no viene del numero de parametros, sino de:

1. Las bandas 1 y >20 son fijas (0.0 delta) en todos los experimentos: band=1 porque user_average_stars ya es el predictor optimo, band=>20 porque el floor del task es muy bajo.
2. La banda 2-5 tiene 23,568 filas pero el ruido intrínseco del usuario con 2-5 reviews no es modelable con correcciones residuales.
3. Solo la banda 6-20 (13,994 filas) contiene residuos aprendibles; su mejora potencial maxima es limitada por el volumen de datos.

**Conclusion:** El camino real hacia MAE < 0.63 es mejorar el incumbent LGBM para la banda 2-5 (el grupo mas grande con mayor MAE actual: 0.716), no añadir capacidad al corrector deep.

---

## Script De Lanzamiento

```bash
cd content-based
uv run python pipelines/deep/train_lightweight_deep.py \
    --max-runs 2 \
    --save-root artifacts/known_user_deep_lightweight_v1 \
    --incumbent-root artifacts/lgbm_raw_router_prefix_deep_v1
```

El script `train_lightweight_deep.py` es un wrapper que llama a `train_known_user_deep.py` con `--config-family v_lightweight` forzado y genera los mismos artefactos que el pipeline completo.

---

## Estado Final

| Run | Decision | Motivo |
|---|---|---|
| runA | `candidate` | Mejor comportamiento en banda 6-20; lightweight viable como alternativa al modelo completo |
| runB | `deprecated` | Sobreregularizado; solo activa banda 1 donde delta=0 |

El artefacto `known_user_deep_lightweight_v1` no genera submission independiente (faltan los artefactos de router completo en el root). Para generar submission habria que re-ejecutar con el pipeline completo o adaptar el script de submission para usar el checkpoint del run.
