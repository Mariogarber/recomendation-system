# Submission 0.93 Analysis

Fecha: `2026-04-10`

Este documento resume por que la submission hibrida bajo de `1.04` a `0.93`, usando solo artefactos ya existentes del deep, del GBM y del blend. No reevalua el leaderboard oculto; la lectura es local y esta orientada a diagnosticar donde se gano y que sigue fallando.

## Fuentes revisadas

- [frozen_embedding_regressor_honest_v1/run_summary.json](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_honest_v1/run_summary.json)
- [frozen_embedding_regressor_honest_v1/ridge_iter03_baseline/band_metrics.csv](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_honest_v1/ridge_iter03_baseline/band_metrics.csv)
- [gbm_regressor_v1_cs30/validation_summary.json](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/gbm_regressor_v1_cs30/validation_summary.json)
- [gbm_regressor_v1_cs30/band_metrics_gbm_raw.csv](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/gbm_regressor_v1_cs30/band_metrics_gbm_raw.csv)
- [gbm_regressor_v1_cs30/band_metrics_blend.csv](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/gbm_regressor_v1_cs30/band_metrics_blend.csv)
- [gbm_regressor_v1_cs30/feature_importance.csv](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/gbm_regressor_v1_cs30/feature_importance.csv)
- [blended_submission_v1/submission_summary.json](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/blended_submission_v1/submission_summary.json)

## Resumen ejecutivo

La mejora no viene de un solo ajuste, sino de dos cosas muy concretas:

1. El GBM cubre el cold start real, que es donde el deep puro no llegaba.
2. El blend mantiene la senal del deep donde hay historial, pero usa el GBM como red de seguridad para los usuarios nuevos.

En validacion, el deep honesto se queda en `MAE = 1.2282` en la referencia Ridge y `1.2302` en el mejor head trainable. El GBM baja a `1.0054` y el blend baja todavia mas a `0.9457`.

## Donde se gana

### 1. El cold start es el gran salto

La banda `0` es la mas importante y la mas dificil. Ahi el deep honesto esta en `MAE = 1.3087`, mientras que el GBM baja a `1.0588`. El blend mantiene exactamente ese mismo nivel en la banda `0`, porque en usuarios nuevos la prediccion final cae al fallback del GBM.

Eso explica buena parte de la subida de calidad total:

- en validacion, el GBM trata `128,830` ejemplos de banda `0`
- esa banda representa el `66.56%` de la validacion del GBM
- en la submission final, el fallback del GBM cubre `169,937` filas, es decir, el `40.97%` del test

Conclusion: la mejora de la submission no se entiende sin el router de cold start. El deep por si solo no resuelve ese tramo.

### 2. El GBM mejora todas las bandas con historial

Comparado con el deep honesto, el GBM reduce MAE en todas las bandas conocidas:

- `1`: de `1.1642` a `0.7945`
- `2-5`: de `1.0849` a `0.7248`
- `6-20`: de `0.9504` a `0.6614`
- `>20`: de `0.9376` a `0.5813`

El recorte relativo esta entre un `19.1%` y un `38.0%`, con la mayor mejora en la banda `>20`.

Esto sugiere que el GBM no solo esta arreglando el cold start, sino que tambien esta haciendo un mejor uso de priors y senales tabulares que el head profundo no explota igual de bien.

### 3. El blend anade una segunda capa de mejora sobre los usuarios conocidos

El blend baja el `MAE` global a `0.9457`. Frente al GBM solo, el salto viene sobre todo de los usuarios con historial:

- `1`: `0.7945`
- `2-5`: `0.7248`
- `6-20`: `0.6614`
- `>20`: `0.5813`

La lectura practica es que el deep aporta algo de refinamiento donde ya habia suficiente contexto, pero el verdadero suelo de calidad lo pone el GBM.

## Que senales esta usando mejor el GBM

La importancia de variables apunta a un modelo dominado por priors y senales de interaccion compactas, no por metadata exotica:

- `user_mean_rating`
- `biz_mean_rating`
- `user_emb_*`
- `user_rating_std`
- `mean_rating_gap`
- `history_log1p`
- `biz_rating_std`

Interpretacion:

- los priors de usuario y negocio son el nucleo
- algunos ejes latentes del embedding todavia ayudan
- la variable `history_log1p` sigue siendo util como proxy de confianza
- las features temporales y de review-context aparecen como soporte, no como motor principal

## Que sigue fallando

### 1. La banda 0 sigue siendo el punto mas debil

Aunque el GBM la mejora mucho, `MAE = 1.0588` sigue siendo claramente peor que el resto de bandas. El modelo sigue teniendo problemas para inferir preferencias cuando no hay historial del usuario.

### 2. La banda 1 sigue siendo inestable

La banda `1` baja mucho con el GBM y el blend, pero sigue siendo una zona fragil. Un solo dato historico no siempre basta para fijar el patron del usuario.

### 3. El deep no gana el partido por si solo

El deep honesto sigue por encima del GBM en `MAE` global. Su mejor papel actual es complementar al tabular, no actuar como modelo unico.

### 4. La mejora depende demasiado de la mezcla de bandas

El salto de calidad no es uniforme: viene de arreglar la cola fria y de preservar la senal en usuarios con mas historial. Eso significa que cualquier cambio en la distribucion de cold start puede mover bastante el resultado final.

## Conclusiones accionables

- Prioridad 1: seguir mejorando el cold start del GBM, porque ahi esta el mayor margen.
- Prioridad 2: probar segmentacion explicita por bandas de historial, especialmente `0` y `1`.
- Prioridad 3: reforzar priors suavizados y estadisticas de negocio/usuario antes de intentar complejidad extra.
- Prioridad 4: tratar el deep como componente de refinamiento para usuarios conocidos, no como solucion principal.

En corto: la bajada de `1.04` a `0.93` parece venir de haber cambiado el foco desde un scorer profundo global a un sistema hibrido con una base tabular mucho mas robusta en cold start. El siguiente salto deberia salir de mejorar la banda `0` y estabilizar la banda `1`.
