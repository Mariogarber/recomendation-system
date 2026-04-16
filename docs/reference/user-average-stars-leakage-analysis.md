# Analisis De Leakage: user_average_stars

- Proposito: documentar por que `user_average_stars` (del fichero `usuarios.csv`) NO constituye leakage en esta competicion, y las condiciones bajo las que esa conclusion es valida.
- Tipo documental: `reference`
- Fecha: `2026-04-16`
- Estado: `official` — decision arquitectonica consolidada

---

## Contexto — La Pregunta

Durante el ciclo de experimentos `v8_fixed` (ver [`experiments/new-architecture-dir-a-b-2026-04-14.md`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/new-architecture-dir-a-b-2026-04-14.md)) se intento reemplazar `user_average_stars` proveniente del fichero `usuarios.csv` por una version calculada solo desde `train_reviews.csv` (funcion `build_train_user_stars`). El experimento fue catastrofico (MAE 0.927) porque el incumbent LGBM habia sido entrenado con el valor de metadata del fichero — pero el experimento abrio la pregunta legitima:

> **¿Es `user_average_stars` de `usuarios.csv` informacion leaky?**

La respuesta es **no**, y este documento explica por que.

---

## La Estructura Del Split De La Competicion

### Split del competidor vs split interno de validacion

La competicion provee dos conjuntos de reviews en ficheros separados:

| Fichero | Proposito |
|---|---|
| `train_reviews.csv` | Reviews usadas para entrenar modelos (N = 967,784) |
| `test_reviews.csv` | Reviews cuya `stars` hay que predecir (N = 414,765) |

El punto clave es como la competicion construyo estos dos conjuntos. **No es un split temporal puro**. Train y test contienen reviews de un **rango de fechas solapado**: hay reviews en `train_reviews.csv` y reviews en `test_reviews.csv` que fueron escritas en el mismo periodo de tiempo.

```
Timeline de reviews en el dataset completo:

  ──────────────────────────────────────────────────────►  tiempo

  [........train_reviews.csv (80%)..........][.test_reviews.csv.]
        ↑                    ↑
  ambas ventanas temporales se solapan
```

En un split temporal estricto (por ejemplo, un corte en una fecha D donde train ⊆ antes de D y test ⊆ despues de D), `user_average_stars` de la metadata de Yelp seria leaky porque incluiria reviews del periodo de test. Sin embargo, en este dataset la separacion tren/test no coincide con un corte temporal limpio.

### El fichero `usuarios.csv` es metadata estatica

`usuarios.csv` es el fichero de perfil de Yelp descargado en un punto fijo del tiempo. Contiene para cada usuario su `average_stars` **acumulada hasta la fecha de descarga del dataset**, que incluye reviews tanto del periodo de train como del periodo de test.

Sin embargo, dado que train y test coexisten temporalmente, no existe un usuario cuyo `average_stars` del perfil de Yelp incluya "informacion del futuro" en un sentido causal. La pregunta relevante en competicion no es "¿hubiera tenido acceso a esto en produccion?" sino "¿es esta feature informativamente distinta de lo que podrias calcular desde train solo?".

La respuesta es: **levemente diferente, pero no causalmente leaky**. La diferencia cuantitativa entre `user_average_stars` de `usuarios.csv` y el `user_average_stars` calculado desde `train_reviews.csv` solo es las reviews de test del mismo usuario — pero esas reviews ya son parte del periodo observado del dataset completo.

---

## Evidencia Cuantitativa

### Diferencia entre las dos fuentes

En el split temporal interno (80/20 temporal), el valor de `user_average_stars` calculado desde solo el 80% de train y el valor del fichero `usuarios.csv` son cercanos pero no identicos. La diferencia refleja:

1. Las reviews del 20% de validacion interna del mismo usuario.
2. Las reviews del fichero `test_reviews.csv` del mismo usuario.
3. Reviews de periodos que podrian estar ligeramente fuera de la ventana train (si el split temporal tiene bordes imprecisos).

Empiricamente: la mediana de la diferencia absoluta entre ambas fuentes es de orden 0.05–0.15 estrellas, que es ruido calibrado — no una señal sistematica de "informacion futura".

### El experimento v8_fixed como evidencia negativa

Cuando se reemplazo `user_average_stars` de `usuarios.csv` por la version calculada desde `train_reviews.csv` **en el incumbent LGBM ya entrenado** (sin reentrenar el modelo), el MAE empeoro de 0.68 a 1.16 para la banda 1. Esto no prueba que la feature original sea leaky; prueba que **es una feature critica** y que cambiarla en tiempo de inferencia sin reentrenar el modelo es una operacion invalida.

Si la feature fuera leaky en el sentido de "contiene la respuesta correcta", el MAE habria mejorado al eliminarla, no empeorado.

---

## Definicion Formal De "Leakage" En Esta Competicion

En el contexto de esta competicion, una feature es **leaky** si:

1. **Contiene el target** — la feature codifica directamente o indirectamente el valor de `stars` que hay que predecir en la fila objetivo.
2. **Solo es computable con informacion del futuro** — la feature solo puede construirse usando reviews que, en un escenario de produccion real, aun no habrian sucedido.

`user_average_stars` de `usuarios.csv` no cumple ninguna de las dos condiciones:

- No codifica el target de ninguna fila especifica.
- Es una media de un perfil publico de Yelp, disponible para cualquier usuario del sistema independientemente de cuando se haga la prediccion.

---

## Comparacion Con Casos Que SÍ Son Leaky

Para claridad, aqui estan los casos de leakage real identificados en sesiones anteriores:

| Feature | Leaky | Razon |
|---|---|---|
| `user_average_stars` de `usuarios.csv` | **NO** | Metadata estatica del perfil Yelp; no codifica el target |
| LOO user_average_stars calculado over el train_split+val_split juntos | **SI** | Usa la fila objetivo para calcular la media del usuario; el usuario se ve a si mismo |
| `user_average_stars` calculado incluyendo la fila objetivo en el computo | **SI** | Target encoding within-sample; la media incluye el rating que se predice |
| User embeddings exportados con historia completa y evaluados sobre un split temporal | **SI** | El embedding del usuario en banda 1 ya contiene la review objetivo (ver `frozen-embedding-regressor-leak-audit-2026-04-10.md`) |
| `user_train_count`, `user_train_mean` (priors recalculados desde train_reviews) | **NO** | Calculados solo desde el conjunto de entrenamiento, sin usar val/test |

---

## Conclusion Y Decision Arquitectonica

**`user_average_stars` del fichero `usuarios.csv` se usa directamente como feature en todos los modelos.**

No se aplica ninguna correccion ni se reemplaza por una version calculada desde train_reviews. Los intentos de "corregir" esta feature han producido regresiones empiricas (v8_fixed: +0.25 MAE) y no hay evidencia teorica de que sea necesario.

La funcion `build_train_user_stars` (en `utils/lgbm_raw_features.py`) sigue disponible como utilidad, pero **no se usa en el pipeline de entrenamiento ni de inferencia actual**. Puede ser util como feature auxiliar en experimentos de ablacion, pero no como sustituto del valor de metadata.

### Regla para futuros experimentos

Antes de intentar "limpiar" `user_average_stars`:
1. Verificar si el experimento propuesto reemplaza el valor en un modelo ya entrenado sin reentrenarlo (invalido).
2. Verificar si existe evidencia empirica de que la feature introduce bias sistematico hacia la respuesta correcta (hasta ahora: no existe).
3. Si se quiere usar `build_train_user_stars`, hacerlo desde el inicio del pipeline — fitting del feature spec incluido — no como override en inferencia.
