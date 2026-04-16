# Señales De Usuario En El Modelo Cold Start

- Proposito: documentar exactamente que informacion de usuario ve el cold start model (rama `cold_model` del router), como se construye cada señal, y que capacidad predictiva aporta.
- Tipo documental: `reference`
- Fecha: `2026-04-16`
- Estado: `official`

---

## Que Es El Cold Start Model

El router de prediccion clasifica cada review de test en una de tres ramas segun el historial del usuario (`history_band`):

| Rama | Condicion | Modelo |
|---|---|---|
| `cold_model` | `history_band == 0` (ningun review del usuario en train) | LightGBM con features de metadata + arquetipos |
| `known_model` | `history_band >= 1` (usuario visto en train) | LightGBM con features de historial |
| `known_prefix_deep_model` | `history_band in [1, 2-5, 6-20]` y deep disponible | LightGBM + deep corrector |

El `cold_model` cubre **128,830 filas de validacion** (el 66% del total de 193,557 filas de test/val frías), y dentro de esas, el desglose por tipo de cold start es:

| Tipo | Filas | % |
|---|---|---|
| Both known (usuario y negocio en train pero sin reviews propias) | 55,511 | 28.7% |
| New user, known item | 113,692 | 58.7% |
| Known user, new item | 9,216 | 4.8% |
| Both new | 15,138 | 7.8% |

La mayoria de filas cold son del tipo "nuevo usuario, negocio conocido" — es decir, el modelo tiene informacion del negocio pero muy poca del usuario.

---

## Grupo 1 — Features Directas De Metadata De Usuario

Estas features provienen de `usuarios.csv` y no requieren historial de reviews del usuario objetivo. Son seguras (ver [`reference/user-average-stars-leakage-analysis.md`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/user-average-stars-leakage-analysis.md)).

| Feature | Fuente | Descripcion |
|---|---|---|
| `user_average_stars` | `usuarios.csv` → `average_stars` | Media acumulada de todas las estrellas del usuario en Yelp. Señal de sesgo del usuario: usuarios con 4.2 tienden a escribir reviews positivas. **Feature mas importante del grupo.** |
| `user_review_count` | `usuarios.csv` → `review_count` | Numero total de reviews escritas por el usuario en toda su historia Yelp. |
| `user_review_count_log1p` | Derivada | `log1p(user_review_count)` — versión comprimida para reducir skew. |
| `user_tenure_days` | Derivada de `yelping_since` y `train_max_timestamp` | Dias desde que el usuario creo su cuenta hasta la fecha del ultimo review de train. Proxy de experiencia. |
| `user_tenure_years` | Derivada | `user_tenure_days / 365.25`. |
| `user_total_votes` | `usuarios.csv` | `useful + funny + cool` recibidos por el usuario. Proxy de engagement comunitario. |
| `user_total_votes_log1p` | Derivada | Version log. |
| `user_engagement_log1p` | Derivada | `log1p(total_votes + fans)`. |
| `user_friends_count` | `usuarios.csv` → `friends` | Numero de amigos en la red Yelp. |
| `user_friends_log1p` | Derivada | Version log. |
| `user_elite_years_count` | `usuarios.csv` → `elite` | Numero de años con estatus elite. Fuerte indicador de usuarios muy activos y comprometidos. |
| `user_elite_any` | Derivada | Flag binario `elite_years_count > 0`. |
| `user_fans` | `usuarios.csv` | Numero de fans. |
| `user_compliment_total` | `usuarios.csv` | Suma de todos los tipos de cumplidos recibidos. |
| `user_compliment_log1p_total` | Derivada | Version log. |
| `user_compliment_nonzero_count` | Derivada | Numero de tipos de cumplido con al menos un recuento. |
| `user_yelping_since_missing` | Derivada | Flag binario: falta la fecha de alta en Yelp. |

**Capacidad predictiva esperada:** Alta para `user_average_stars` (sesgo individual), baja para el resto sin combinarse con arquetipos. Un usuario nuevo que tiene `average_stars=4.5` predeciblemente dara 4–5 estrellas; uno con `average_stars=2.1` dara 1–2.

---

## Grupo 2 — Arquetipos De Usuario (K-Means)

El cold model no puede usar el historial de reviews del usuario porque no existe. Para suplir esa ausencia, el sistema construye **64 arquetipos de usuario** via K-Means clustering sobre las features numericas del grupo 1.

### Como se construyen los arquetipos

```python
ROUTER_ARCHETYPE_NUMERIC_COLUMNS = [
    "user_average_stars",        # sesgo rating
    "user_review_count_log1p",   # actividad
    "user_tenure_years",         # experiencia
    "user_total_votes_log1p",    # impacto comunitario
    "user_engagement_log1p",     # engagement (votos + fans)
    "user_friends_log1p",        # red social
    "user_elite_years_count",    # compromisos con la plataforma
    "user_compliment_log1p_total", # calidad percibida de reviews
    "user_compliment_nonzero_count", # diversidad de reconocimiento
]
```

Proceso:
1. Las 9 features se normalizan (z-score con media/std del conjunto de entrenamiento).
2. Se aplica `MiniBatchKMeans(n_clusters=64)`.
3. A cada usuario se le asigna el archetype ID del centroide mas cercano (`user_archetype_id`).
4. Si el usuario tiene mas del 25% de features faltantes se le asigna la etiqueta especial `__metadata_sparse__` en vez de un archetype.

### Features derivadas del arquetipo

| Feature | Descripcion |
|---|---|
| `user_archetype_id` | Etiqueta categorica del arquetipo (`archetype_000` … `archetype_063` o `__metadata_sparse__`). Feature categorica que el LightGBM puede usar directamente via split categorico. |
| `user_archetype_distance` | Distancia euclidiana al centroide del arquetipo asignado (en espacio normalizado). Users lejos del centroide son mas atipicos dentro de su grupo. |
| `user_metadata_completeness` | Fraccion de las 9 features de clustering que no son NaN (0.0–1.0). |
| `user_metadata_sparse_flag` | Flag 1.0 si `completeness < 0.75`. |
| `user_activity_bucket` | Bucket categorico de `user_review_count`: `"0"`, `"1"`, `"2-5"`, `"6-20"`, `"21-100"`, `">100"`. |
| `user_reputation_bucket` | Bucket categorico de `user_average_stars`: `"1.0-2.5"`, `"2.5-3.5"`, `"3.5-4.0"`, `"4.0-4.5"`, `"4.5-5.0"`. |
| `user_tenure_bucket` | Bucket categorico de `user_tenure_years`: `"<1y"`, `"1-3y"`, `"3-6y"`, `">6y"`. |

**Capacidad predictiva:** Los arquetipos permiten al modelo capturar patrones como "usuarios de este perfil (alta actividad, rating medio 4.2, muchos amigos) tienden a dar ~4 estrellas en negocios de tipo X". Sin el historial individual es la mejor aproximacion disponible.

---

## Grupo 3 — Priors De Arquetipo Cruzados Con Features Del Negocio

Estas son las features mas sofisticadas del cold model. Para cada combinacion de `(user_archetype_id, dimension_del_negocio)` se calcula una **media de rating suavizada** (Bayesian smoothing con prior = global mean).

### Formula de suavizado

$$\mu_{\text{smoothed}} = \frac{\sum_i r_i + \alpha \cdot \mu_{\text{global}}}{n + \alpha}$$

donde $\alpha$ es el parametro de suavizado (varia por dimension: 20 para overall, 50 para dimensiones mas granulares).

### Tablas de priors de arquetipo

| Feature base | Cruce | Alpha | Descripcion |
|---|---|---|---|
| `archetype_train_mean` | `user_archetype_id` | 20 | Media suavizada de rating del arquetipo en todo el conjunto de entrenamiento |
| `archetype_train_support_count` | `user_archetype_id` | — | Numero de reviews del arquetipo en train |
| `archetype_train_bias` | Derivada | — | `archetype_train_mean - global_mean` |
| `archetype_state_mean` | `(archetype, business_state)` | 50 | Rating medio del arquetipo para negocios en ese estado |
| `archetype_state_support_count` | — | — | Soporte |
| `archetype_city_mean` | `(archetype, business_city_top)` | 50 | Rating medio del arquetipo para negocios en esa ciudad (top 100 ciudades) |
| `archetype_city_support_count` | — | — | Soporte |
| `archetype_star_bin_mean` | `(archetype, business_star_bin)` | 50 | Rating medio del arquetipo para negocios en ese rango de estrellas (`1.0-2.5`, `2.5-3.5`, …) |
| `archetype_star_bin_support_count` | — | — | Soporte |
| `archetype_open_mean` | `(archetype, business_is_open)` | 50 | Rating medio del arquetipo para negocios abiertos/cerrados |
| `archetype_open_support_count` | — | — | Soporte |
| `archetype_category_mean` | `(archetype, business_primary_category_family)` | 50 | Rating medio del arquetipo para negocios de esa categoria principal (top 32 categorias) |
| `archetype_category_support_count` | — | — | Soporte |

### Gaps derivados de los priors

| Feature | Formula | Interpretacion |
|---|---|---|
| `archetype_business_star_gap` | `archetype_train_mean - business_stars` | Diferencia entre el nivel de rating del arquetipo y la calidad media del negocio. Positivo = arquetipo mas "generoso" que la media del negocio. |
| `archetype_state_gap` | `archetype_state_mean - business_stars` | Preferencia geografica del arquetipo vs calidad del negocio. |
| `archetype_category_gap` | `archetype_category_mean - business_stars` | Preferencia por categoria vs calidad del negocio. |

**Capacidad predictiva:** Las priors cruzadas son la señal mas rica disponible para cold users. Un usuario cold de arquetipo "foodie activo, alta reputacion" puede cruzarse con "categoria = Restaurants" → el sistema sabe que ese perfil tiende a dar 3.9 a restaurantes en media, aunque nunca haya visitado este restaurante especifico.

---

## Grupo 4 — Features De La Red Social (Friends)

Estas features aprovechan la lista de amigos del usuario (campo `friends` en `usuarios.csv`). Requieren construir una tabla de lookup `(reviewer_user_id, business_id, stars)` sobre `train_reviews.csv`.

| Feature | Descripcion |
|---|---|
| `friend_business_mean` | Media de las estrellas dadas por los amigos del usuario a este negocio en train. Si ningun amigo ha revisado el negocio, se imputa con `global_mean`. |
| `friend_business_count_log1p` | `log1p(numero de amigos que han revieweado el negocio)`. Señal de confianza: cuantos mas amigos, mas fiable la media. |
| `friend_business_bias` | `friend_business_mean - global_mean`. Cuanto por encima o debajo del promedio global han puntuado los amigos del usuario este negocio. |

**Disponibilidad:** Solo tiene señal cuando (a) el usuario tiene amigos en la lista, y (b) al menos uno de esos amigos ha revieweado el negocio en train. Para la mayoria de cold users (nuevos usuarios) la lista de amigos puede estar vacia o sus amigos no han revieweado el negocio en concreto.

**Capacidad predictiva esperada:** Alta cuando disponible (correlacion social de ratings), baja cuando no hay cobertura. Cae a `global_mean` en ausencia de cobertura → neutro, no perjudicial.

---

## Grupo 5 — Features Del Negocio (Informacion Disponible Para Todos Los Casos)

El cold model tambien tiene acceso a toda la informacion del negocio. Estas features no son especificas de señales de usuario pero condicionan la prediccion del cold model:

| Subcategoria | Features principales |
|---|---|
| Metadata Yelp | `business_stars`, `business_review_count`, `business_review_count_log1p`, `business_rating_per_review` |
| Priors de train | `business_train_mean_rating`, `business_train_rating_std`, `business_train_vs_yelp_gap` |
| Estructura | `business_attributes_count`, `business_attribute_true_count`, `business_categories_count`, `business_is_open` |
| Geografica | `business_city_top`, `business_state`, `business_primary_category_family`, `business_star_bin`, `business_latitude`, `business_longitude` |
| Temporal | `review_days_since_train_start`, `review_days_since_train_end` (positivo = la review es posterior al corte de train) |

---

## Resumen: Que Ve El Cold Model Para Un Usuario Sin Historial

Para un usuario completamente nuevo (tipo `both_new` o `new_user_known_item`), el cold model dispone de:

```
Usuario nuevo → Señales disponibles:
  ├── user_average_stars              ← Sesgo individual del usuario
  ├── user_review_count, tenure, etc. ← Nivel de experiencia
  ├── user_archetype_id               ← Perfil de comportamiento de rating
  │     ├── archetype_train_mean      ← Rating típico de su perfil
  │     ├── archetype_category_mean   ← Preferencia por tipo de negocio
  │     ├── archetype_state_mean      ← Preferencia geográfica
  │     └── archetype_star_bin_mean   ← Preferencia por calidad del negocio
  ├── friend_business_mean (si aplica)← Señal social de la red
  └── [todas las features del negocio]← Calidad y perfil del negocio objetivo
```

La señal de usuario mas fuerte para un usuario nuevo es `user_average_stars` combinada con `user_archetype_id`. La combinacion permite al modelo distinguir entre:

- Usuario casual, rating_mean=4.3, categoria=Restaurants → probablemente dara 4 estrellas
- Usuario elite, rating_mean=3.1, critico → probablemente dara 2–3 estrellas
- Usuario nuevo sin metadata (`__metadata_sparse__`) → fallback al prior global y priors del negocio

---

## Limitaciones Y Ceiling Del Cold Model

El ceiling de MAE del cold model sobre la banda 0 es aproximadamente **0.63** (observado: 0.6328 en `lgbm_raw_router_prefix_deep_v1`, 0.6254 en `lgbm_feature_first_short_router_v1_gpu`). Las razones son estructurales:

1. **Sin historial propio**, `user_average_stars` es la mejor señal individual disponible pero tiene ruido intrinseco: el rating de una review especifica depende del estado de animo del usuario, la calidad real del servicio ese dia, y muchos factores no observables.

2. **Los arquetipos capturan el comportamiento medio del perfil** pero no la varianza individual. Dos usuarios con el mismo arquetipo pueden tener distribuciones de rating completamente distintas.

3. **La cobertura de amigos es baja** para usuarios nuevos: si el usuario acaba de unirse a Yelp, su lista de amigos puede estar vacia.

4. **El 58.7% de filas cold son `new_user_known_item`** — el negocio tiene historial en train (y sus features son ricas), pero el usuario es completamente nuevo. En este caso la prediccion colapsa efectivamente a: "¿cual es el rating tipico de negocios de este tipo para usuarios de este perfil?", que es un prior con ruido moderado.

La unica via conocida de mejora significativa del cold model es incorporar señales del negocio mas ricas que las actuales (como embeddings semanticos de contenido del negocio — explorado en `lgbm_router_v6` con delta cold MAE = −0.027).
