# Guía de interpretación del reporte de calidad de embeddings

Este documento explica, sección por sección, el HTML generado por [`content-based/analyze_embeddings_report.py`](../content-based/analyze_embeddings_report.py) y guardado en `content-based/artifacts/competition_embeddings_v1/report_final/embedding_quality_report.html`.

La idea es que el informe no se lea solo como un conjunto de tablas bonitas, sino como una herramienta para decidir si los embeddings sirven para competir, dónde fallan y qué conviene mejorar.

## 1. Resumen Ejecutivo

Esta primera zona condensa las conclusiones más importantes del reporte:

- `Negocios con embedding completo`: número de negocios cubiertos por `business_full_features`.
- `Usuarios con embedding manual`: número de usuarios cubiertos por el embedding manual.
- `Usuarios con embedding profundo`: número de usuarios cubiertos por `user_deep_features`.
- `Usuarios del snapshot con embedding profundo`: cobertura real sobre el universo de usuarios del snapshot.
- `MAE temporal oficial deep`: métrica principal de validacion del modelo profundo de usuario.
- `Usuarios history-based`: cuantos usuarios del embedding profundo se construyeron usando historial real, no solo metadata.
- `Uplift social deep`: diferencia media de similitud entre amigos y no-amigos emparejados.

### Como interpretarlo

El resumen ejecutivo sirve para responder tres preguntas:

1. El espacio cubre suficiente poblacion para ser util?
2. El modelo profundo mejora la prediccion de rating?
3. La geometria aprendida tiene señales semanticas o sociales coherentes?

### Errores frecuentes

- Confundir cobertura con calidad. Tener muchas filas no implica que el embedding sea bueno.
- Leer el `MAE` oficial como si fuera comparable directamente con la tabla `Utilidad`. No lo es: esa tabla mezcla una validacion interna con un scorer lineal post-export.
- Tomar el `uplift social` como causalidad social. Solo indica homofilia asociativa, no que ser amigo provoque similitud.

### Decision practica

Si el `MAE` es razonable y la cobertura es alta, el embedding merece seguir al siguiente paso de competicion. Si la cobertura cae mucho en `metadata_only`, hay que reforzar cold-start.

## 2. Cobertura y Salud del Espacio

Esta seccion explica si los embeddings estan completos, balanceados y sanos desde el punto de vista numerico.

### Que mide

- `Fuente del embedding por usuario`: cuantos usuarios vienen de `history`, `metadata_only` o `default_only`.
- `Bandas de historial`: distribucion de usuarios por cantidad de reviews en train.
- `history_band x embedding_source`: mezcla entre historial y fallback.
- `Bloques del embedding completo de negocio`: resumen por bloques del vector manual de negocio.
- `Salud de las representaciones`: estadisticas de densidad, normas y dimensiones muertas.

### Como leer las metricas de salud

Para espacios densos como `business_deep` y `user_deep`, mira:

- `norm_mean` y `norm_std`: si son muy extremas, el espacio puede estar mal escalado.
- `zero_norm_count`: valores altos indican embeddings vacios o rotos.
- `dead_dimension_count`: dimensiones casi constantes.
- `random_pair_cosine_mean`: si es demasiado alto, el espacio esta colapsando y muchos vectores se parecen sin razon.
- `pca_explained_variance`: si muy pocas componentes explican casi todo, el espacio puede ser demasiado anisotropo.

Para espacios dispersos como `business_full_features` y `user_profile_features`, mira:

- `density`: proporción de celdas no nulas.
- `row_nnz_mean`: cuantos bloques reales tiene cada fila.
- `norm_mean`: magnitud media del vector.

### Errores frecuentes

- Confundir `metadata_only` con fracaso absoluto. En realidad es un fallback correcto para cold-start, pero hay que medirlo aparte.
- Pensar que un espacio denso siempre es mejor. Un espacio denso puede estar peor estructurado que uno disperso, aunque sea mas cómodo para un modelo.
- Comparar directamente densidad de un espacio disperso con uno denso. No tiene sentido.

### Decision practica

Si la cobertura de `history` es alta y el número de `default_only` es casi nulo, el pipeline es utilizable. Si hay muchas dimensiones muertas o normas degeneradas, hay que revisar entrenamiento, normalizacion o exportacion.

## 3. Utilidad Para La Tarea Final

Esta es una de las secciones mas importantes porque conecta los embeddings con la competicion.

### Que incluye

La tabla `Comparativa de scorers` compara tres espacios:

- `manual_profile + business_full`
- `user_deep + business_deep`
- `deep_user_encoder original`

Las metricas son:

- `MAE`
- `RMSE`

La tabla `AUC de preferencia` mide, para cada usuario, si el modelo ordena mejor ratings altos frente a ratings bajos.

La tabla `MAE por banda de historial` separa el rendimiento por:

- `0`
- `1`
- `2-5`
- `6-20`
- `>20`

### Como interpretarlo

- Si `user_deep + business_deep` baja el `MAE`, el espacio aprendido sirve para scoring real.
- Si el `AUC` mejora, el embedding no solo ajusta valores medios, sino tambien el orden relativo de preferencias.
- Si el rendimiento sube solo en usuarios con mucho historial, el modelo esta aprendiendo bien, pero aun no resuelve cold-start.

### Errores frecuentes

- Leer `AUC` como si fuera la metrica principal de la competicion. Es una señal auxiliar, no la metrica objetivo.
- Comparar modelos sin estratificar por historial. El promedio global puede esconder que el modelo solo funciona en usuarios con muchas reviews.
- Confundir un mejor `MAE` en el scorer lineal post-export con una mejora del entrenamiento original. Son dos diagnósticos distintos.

### Decision practica

Si el espacio profundo mejora claramente sobre el manual, tiene sentido usarlo como base del scorer final. Si el manual sigue igual o mejor en varios cortes, conviene mantener ambos como ensemble o baseline alternativo.

## 4. Embeddings de Negocio

Esta seccion evalua si los negocios cercanos en el espacio realmente tienen sentido semantico.

### Coherencia local global

La tabla `business_coherence_overall.csv` resume, para cada espacio:

- `same_category_ratio`
- `same_city_ratio`
- `category_entropy`

Interpretacion:

- `same_category_ratio`: proporción de vecinos top-k que comparten la categoria principal del negocio ancla.
- `same_city_ratio`: proporción de vecinos que estan en la misma ciudad.
- `category_entropy`: diversidad de categorias entre los vecinos. Menor suele significar vecinos mas homogeneos.

### Coherencia por categoria

La tabla `business_coherence_by_category.csv` repite esas metricas por categoria ancla.

Sirve para detectar:

- categorias faciles, donde la vecindad es muy estable
- categorias mezcladas, donde el embedding capta más mezcla semantica
- categorias dominantes como `Restaurants`, que suelen salir mejor por volumen

### Mapa PCA de negocios profundos

El scatter PCA enseña una proyeccion 2D del espacio `business_deep`.

Sirve para ver:

- agrupamientos gruesos
- separacion entre categorias
- posibles colapsos o superposiciones

Pero no debe leerse como una prueba formal de calidad, porque PCA solo conserva una parte de la varianza.

### Similitud entre centroides de categorias

La heatmap compara los centroides medios de cada categoria.

Interpretacion:

- diagonal alta: cada categoria tiene identidad propia
- bloques altos fuera de diagonal: categorias afines o muy mezcladas

### Errores frecuentes

- Pensar que menor diversidad siempre es mejor. Si la diversidad es demasiado baja, el embedding se vuelve rígido y poco util para recomendacion.
- Confundir categorias cercanas con error. A veces dos categorias cercanas de verdad comparten patrones de consumo.
- Sobreinterpretar el PCA: un buen o mal dibujo 2D no garantiza nada por si solo.

### Decision practica

Si `business_full` gana en coherencia pura pero `business_deep` es razonable, se puede usar `business_full` para explicabilidad y `business_deep` para scoring. Si `business_deep` pierde demasiado en coherencia, conviene revisar su aprendizaje.

## 5. Embeddings de Usuario

Esta seccion compara el espacio manual y el profundo, y luego segmenta usuarios.

### Consistencia manual vs deep

La comparativa reporta:

- `Correlation of distances`
- `Mean top-10 neighbor overlap`
- numero de usuarios analizados

Interpretacion:

- correlacion alta significa que ambos espacios ordenan usuarios de forma parecida
- overlap alto significa que sus vecinos principales coinciden
- si la correlacion es moderada pero el deep mejora en utilidad, el espacio profundo esta capturando señales nuevas, no solo copiando el manual

### Clustering historico

Se aplica a usuarios con historial real relevante. La tabla incluye:

- `n_users`
- `history_median`
- `tenure_mean`
- `elite_rate`
- `fans_mean_z`
- `social_capital_mean_z`
- `cluster_label`

`silhouette` mide separacion de clusters:

- cerca de 1: clusters muy separados
- cerca de 0: clusters solapados
- negativo: mala particion

### Clustering cold-start

Se aplica a usuarios con `metadata_only` o `default_only`.

Sirve para entender si la metadata ya separa perfiles sociales o demograficos aunque no haya historial.

### Errores frecuentes

- Mezclar `histórico` y `cold-start` en un unico cluster. Eso suele contaminar la interpretacion.
- Pensar que clusters bonitos implican mejor recomendacion. Los clusters solo ayudan a entender el espacio.
- Interpretar `elite_rate` o `fans` como causa unica de pertenencia. Son rasgos descriptivos, no explicaciones completas.

### Decision practica

Si el clustering historico produce segmentos interpretable y el cold-start no es caotico, el espacio es util para perfilado, fallback y analitica de producto.

## 6. Amistades y Homofilia Social

Esta parte mide si los usuarios que son amigos aparecen mas cerca en el espacio que usuarios comparables que no lo son.

### Que mide

La seccion incluye:

- `Usuarios con amigos válidos`
- `Aristas dirigidas válidas`
- `p90 grado social`
- `máximo grado`
- resumen friend vs matched non-friend
- subconjunto `history-history`
- visualizacion del uplift social

### Como se construye la comparacion

El script no compara amigos contra cualquier no-amigo al azar. Hace matching por:

- tipo de embedding
- actividad
- tenure
- fans

Eso reduce sesgo por popularidad, actividad o cobertura.

### Como leer las metricas

- `friend_mean_cos`: similitud media entre amigos.
- `matched_non_friend_mean_cos`: similitud media entre no-amigos emparejados.
- `uplift`: diferencia entre ambas.
- `paired_win_rate`: porcentaje de pares donde el amigo gana al no-amigo emparejado.

La lectura correcta es:

- uplift positivo = homofilia social asociativa
- uplift pequeño pero estable = señal real, aunque no muy fuerte
- uplift sin control de actividad no valdria como conclusión

### Errores frecuentes

- Tomar todos los amigos como si fueran comparables entre si. No lo son: hay usuarios muy conectados y usuarios casi aislados.
- Ignorar el `history-history`. Ese subconjunto suele ser el mas limpio.
- Convertir la homofilia en causalidad. Los embeddings reflejan similitud; no prueban influencia.

### Decision practica

Si el uplift es positivo, la señal social puede entrar como feature auxiliar, regularizacion o analitica de segmentacion. No suele justificar por si sola un modelo puramente social.

## 7. Recomendaciones Para Competición

Esta seccion no diagnostica; traduce el informe a accion.

Las recomendaciones actuales del HTML apuntan a:

- usar `user_deep + business_deep` como espacio principal
- conservar `manual_profile + business_full` como baseline y explicabilidad
- separar `cold-start`, `single-review` e `historical`
- usar la similitud social como señal auxiliar, no como unico criterio
- atacar sobre todo el cold-start y el scorer final

### Como interpretarlas

No son reglas fijas, sino consecuencias de las metricas anteriores. Si una seccion cambia, estas recomendaciones deben reevaluarse.

## 8. Lectura Rápida Recomendada

Si quieres evaluar el reporte en cinco minutos, sigue este orden:

1. `Resumen Ejecutivo`
2. `Utilidad Para La Tarea Final`
3. `Cobertura y Salud del Espacio`
4. `Embeddings de Negocio`
5. `Embeddings de Usuario`
6. `Amistades y Homofilia Social`

## 9. Archivo de apoyo

El HTML final se genera desde:

- [`content-based/analyze_embeddings_report.py`](../content-based/analyze_embeddings_report.py)

Y los artefactos de salida viven en:

- `content-based/artifacts/competition_embeddings_v1/report_final/`

Si una seccion cambia en el script, esta guia deberia actualizarse para conservar la correspondencia uno a uno entre metricas y significado.
