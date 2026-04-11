# Guia de features del sistema content-based

Esta guia documenta como se construyen y como se deben interpretar las features del modulo [`content-based`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based).
Hoy conviene leerla con tres ideas en mente:

- existe una familia manual de usuario
- existe una familia profunda de usuario ya implementada
- el reporte de embeddings es diagnostico, no baseline final

El objetivo es que puedas entender:

- que representa cada bloque de features de negocio
- que representa cada bloque de features de usuario
- como se calculan
- como activar o desactivar partes de la representacion
- como manipular las matrices sin mezclar senal segura con variables con riesgo de leakage
- como interpretar los defaults reales de los builders y del runner de competicion

## 1. Idea general del pipeline

El pipeline actual separa claramente dos niveles:

1. Primero construye una representacion de cada negocio a partir de su metadata y de priors derivados solo de `train_reviews.csv`.
2. Despues construye una representacion de cada usuario agregando los vectores de los negocios que ha valorado en train.

Eso produce dos familias de features:

- `business features`: describen al item en si
- `user features`: describen el gusto del usuario y su metadata segura

La construccion esta implementada en:

- [`content-based/utils/business_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/business_representation.py)
- [`content-based/utils/user_representation.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/user_representation.py)
- [`content-based/utils/deep_user_embeddings.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/deep_user_embeddings.py)
- [`content-based/utils/business_features.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/business_features.py)

## 2. Features de negocio

La representacion de negocio se divide en cinco bloques:

- `geo`
- `categories`
- `attributes`
- `hours`
- `priors`

Y se expone en tres matrices:

- `content_matrix`: solo contenido puro del negocio
- `prior_matrix`: solo priors agregados desde train
- `full_matrix`: concatenacion de `content_matrix + prior_matrix`

En el smoke test actual la dimensionalidad es:

- `geo`: 84 columnas
- `categories`: 558 columnas
- `attributes`: 225 columnas
- `hours`: 4 columnas
- `priors`: 5 columnas
- `content_matrix`: 871 columnas
- `full_matrix`: 876 columnas

### 2.1 Bloque `geo`

Este bloque captura localizacion y estado operativo del negocio.

Incluye:

- `geo__state_*`: one-hot del estado
- `geo__city_*` o `geo__city_hash__*`: representacion de la ciudad
- `geo__latitude_z`: latitud normalizada con z-score
- `geo__longitude_z`: longitud normalizada con z-score
- `geo__is_open`: indicador numerico de apertura

#### Como se calcula

- `state` y `city` se limpian y los faltantes pasan a `__unknown__`.
- Si una ciudad aparece menos de `min_city_freq`, se manda a `__other__`.
- Si el numero de ciudades supervivientes no supera `max_city_ohe`, la ciudad va en one-hot.
- Si lo supera, se usa hashing con `city_hash_dim`.
- `latitude` y `longitude` se imputan con la mediana y luego se normalizan.
- `is_open` se convierte a numerico y los faltantes se rellenan con `0`.

#### Como interpretarlo

- Las columnas `geo__state_*` y `geo__city_*` indican presencia geografica.
- Las columnas `geo__city_hash__*` no son interpretables individualmente; sirven para retener senal de ciudad sin explotar dimensionalidad.
- `geo__latitude_z` y `geo__longitude_z` conservan posicion relativa continua.
- `geo__is_open` permite distinguir negocios activos o cerrados.

#### Como manejarlo

- Si quieres interpretabilidad, intenta forzar `one_hot` subiendo `max_city_ohe` o relajando `min_city_freq`.
- Si quieres robustez con muchas ciudades, manten hashing.
- Si te preocupa la mezcla de localizacion fina y coarse, puedes usar solo `state` y quitar ciudad usando `business_blocks`.

### 2.2 Bloque `categories`

Este bloque recoge las categorias textuales del negocio como presencia o ausencia.

Ejemplos de nombres:

- `category__restaurants`
- `category__american_traditional`
- `category__nightlife`

#### Como se calcula

- La columna `categories` se separa por comas.
- Se eliminan vacios y duplicados dentro de cada negocio.
- Se cuentan frecuencias globales en el dataset de negocios.
- Solo se conservan categorias con frecuencia al menos `min_category_freq`.
- La codificacion final es multi-hot binaria.

#### Como interpretarlo

- Cada columna vale `1` si el negocio contiene esa categoria, `0` si no.
- No hay peso TF-IDF ni frecuencia interna; es una presencia binaria.
- Es uno de los bloques mas interpretables y mas seguros del sistema.

#### Como manejarlo

- Si quieres mas granularidad, baja `min_category_freq`.
- Si el vector se hace demasiado grande o ruidoso, sube `min_category_freq`.
- Para ablations, este bloque suele ser buen candidato para medir cuanto aporta semantica del negocio frente a geo u horarios.

### 2.3 Bloque `attributes`

Este bloque representa metadata estructurada del negocio que viene en `attributes`.

Ejemplos habituales:

- `attribute__restaurantsdelivery_true`
- `attribute__bikeparking_false`
- `attribute__businessparking_garage_true`
- `attribute__ambience_casual_true`
- `attribute__noiselevel_loud`

#### Como se calcula

- La columna `attributes` se parsea como diccionario.
- Si hay subdiccionarios, se aplanan con claves tipo `BusinessParking.garage`.
- Los booleanos se normalizan a `true` y `false`.
- Cada feature es un par `clave=valor`.
- Solo se conservan claves con un numero de valores manejable:
  - si una clave tiene mas de `max_attribute_values_per_key`, se excluye entera
- Dentro de las claves elegibles:
  - si un valor aparece al menos `min_attribute_value_freq`, se mantiene como columna propia
  - si no llega al minimo, puede caer en un bucket `other` de esa clave
- La codificacion final tambien es multi-hot binaria.

#### Como interpretarlo

- Cada columna indica que ese negocio tiene exactamente ese valor para ese atributo.
- Este bloque mezcla booleanos, nominales y algunas cantidades discretizadas como texto.
- Es bastante interpretable mientras no se abuse del bucket `other`.

#### Como manejarlo

- Si quieres atributos mas finos, baja `min_attribute_value_freq`.
- Si quieres evitar explosion de cardinalidad, baja `max_attribute_values_per_key`.
- Si notas mucho ruido semantico, revisa `feature_metadata.csv` y `clean_business_table.parquet` para ver que pares clave-valor han sobrevivido realmente.

### 2.4 Bloque `hours`

Este bloque resume los horarios del negocio en cuatro features numericas compactas:

- `hours__open_days_count_z`
- `hours__weekly_open_minutes_z`
- `hours__weekend_days_open_z`
- `hours__late_night_days_z`

#### Como se calcula

- La columna `hours` se parsea como diccionario `dia -> intervalo`.
- Para cada negocio se extraen:
  - numero de dias abiertos
  - minutos totales abiertos por semana
  - numero de dias abiertos en fin de semana
  - numero de dias con cierre tardio
- Si un intervalo termina antes de empezar, se asume que cruza medianoche.
- Si inicio y fin coinciden, se interpreta como apertura 24 horas.
- Si faltan horarios o son invalidos, el negocio recibe ceros en bruto.
- Despues cada una de las cuatro variables se normaliza con z-score.

#### Como interpretarlo

- Son features continuas, no binarias.
- Valores positivos indican negocios por encima de la media global en esa dimension.
- `late_night_days` captura bien restaurantes, bares o servicios nocturnos.

#### Como manejarlo

- Es un bloque pequeno y barato; suele merecer la pena mantenerlo.
- Si quieres mas interpretabilidad, consulta los valores crudos en `clean_business_table`.
- Si sospechas que la normalizacion global perjudica segmentos concretos, este bloque es facil de reescalar o sustituir.

### 2.5 Bloque `priors`

Este bloque no es metadata directa del CSV de negocios, sino agregados calculados solo con `train_reviews.csv`.

Incluye:

- `prior__seen_in_train`
- `prior__train_review_count_log1p`
- `prior__train_support_percentile`
- `prior__train_average_stars`
- `prior__train_rating_std`

#### Como se calcula

- Se agrupan las reviews de train por `business_id`.
- Se recomputan:
  - numero de reviews en train
  - media de rating en train
  - desviacion tipica del rating en train
- Se deriva tambien:
  - `log1p(review_count)`
  - soporte relativo respecto al maximo del dataset
  - flag de si el negocio se ha visto o no en train
- Si un negocio no aparece en train, los agregados se rellenan con `0` y `seen_in_train=0`.

#### Por que existe separado

Este bloque se separa del contenido puro porque:

- aporta senal predictiva fuerte
- pero no describe el contenido intrinseco del negocio
- y exige control por riesgo de leakage

El codigo excluye expresamente usar `stars` y `review_count` crudos de `negocios.csv` como priors directos.

#### Como manejarlo

- Usa `content_matrix` si quieres un perfil semantico puro del item.
- Usa `full_matrix` si quieres mezclar contenido con popularidad o calidad observada en train.
- Usa `prior_matrix` para hacer ablations o para un baseline solo de priors.
- Si quieres una evaluacion estricta de cold start de item, este bloque debe vigilarse mucho.

## 3. Features de usuario

La representacion de usuario se divide en dos bloques:

- `profile`
- `metadata`

Y se expone en tres matrices:

- `profile_matrix`
- `metadata_matrix`
- `full_user_matrix`

La idea principal es:

- `profile`: resume gustos agregando los negocios que el usuario ya valoro
- `metadata`: anade metadata segura del usuario, separada del gusto

En el smoke test actual:

- `profile_matrix`: 871 columnas
- `metadata_matrix`: 18 columnas
- `full_user_matrix`: 889 columnas

### 3.1 Bloque `profile`

Cada feature de `profile` es la version agregada de una feature de negocio.

Ejemplos:

- `profile__category__restaurants`
- `profile__attribute__bikeparking_true`
- `profile__hours__weekly_open_minutes_z`
- `profile__geo__state_NV`

#### Que significa

No significa que el usuario tenga esa categoria, sino que:

- el usuario ha interactuado con negocios que tienen esa propiedad
- y la intensidad de esa componente depende del modo de agregacion

Es decir, `profile__category__mexican` alto implica afinidad del usuario con negocios mexicanos segun sus ratings en train.

#### Como se calcula

1. Se elige una vista del negocio:
   - `content`
   - `prior`
   - `full`
   - o un subconjunto concreto de bloques con `business_blocks`
2. Cada review del usuario hereda el vector del negocio valorado.
3. Ese vector se pondera segun `aggregation_mode`.
4. Se hace una media ponderada por usuario.

#### Modos de agregacion soportados

##### `mean`

- todas las interacciones pesan lo mismo
- es la opcion mas estable y simple

Util cuando:

- quieres un perfil promedio sin introducir intensidad por rating

##### `rating`

- cada negocio pesa su rating observado

Util cuando:

- quieres que 5 estrellas cuenten mas que 2 estrellas

Riesgo:

- mezcla sesgo de escala del usuario con preferencia

##### `centered`

- peso por interaccion: `rating - media_del_usuario_en_train`

Es el modo por defecto porque:

- modela preferencia relativa del usuario
- reduce el sesgo de usuarios que puntuan todo alto o todo bajo

Fallos controlados por el codigo:

- si el usuario tiene una sola review, se usa peso `1`
- si la suma de pesos absolutos queda a cero, tambien se cae al peso `1`

Esto es importante en tu dataset porque la mayoria de usuarios tienen poquisimo historial.

##### `recency`

- pondera mas fuerte las reviews recientes
- usa un decaimiento exponencial con `recency_half_life_days`

Util cuando:

- quieres modelar cambio de gustos en el tiempo

Si faltan timestamps, cae a pesos uniformes.

#### Como interpretarlo

- Si usas `business_view="content"`, el perfil del usuario vive en el mismo espacio semantico que los items.
- Eso permite usar similitud directa usuario-item.
- Si usas `business_view="full"`, el perfil del usuario mezcla gustos semanticos y exposicion a items populares o bien valorados.

#### Defaults reales

- el builder manual por defecto usa `business_view="content"`
- la corrida de competicion usa `business_view="full"` para el bundle manual
- el `feature_metadata` y `user_profile_summary` ya registran la vista real usada en la invocacion, incluyendo `business_view`, `business_blocks` y `profile_source`

#### Como manejarlo

- Para recomendacion content-based pura, empieza con `business_view="content"`.
- Para mejorar prediccion de rating, luego compara contra `full`.
- Si quieres entender el aporte de cada familia, usa `business_blocks=["categories"]`, `["attributes"]`, `["geo"]`, etc.
- Si quieres construir un scorer interpretable, `centered + content` es la combinacion mas limpia.

### 3.2 Bloque `metadata`

Este bloque contiene metadata del usuario que se considera relativamente segura y separada del gusto inferido.

Incluye:

- `metadata__tenure_days_z`
- `metadata__elite_years_count_z`
- `metadata__elite_any`
- `metadata__useful_log1p_z`
- `metadata__funny_log1p_z`
- `metadata__cool_log1p_z`
- `metadata__fans_log1p_z`
- `metadata__compliment_*_log1p_z`

#### Como se calcula

- `yelping_since` se transforma en antiguedad en dias tomando como referencia el final temporal de train.
- `elite` se convierte en:
  - numero de anos elite
  - flag binario de si alguna vez fue elite
- `useful`, `funny`, `cool`, `fans` y `compliment_*`:
  - se convierten a numerico
  - se les aplica `log1p`
  - despues se normalizan con z-score
- Los faltantes se rellenan.

#### Que metadata se excluye explicitamente

No se usan en V1:

- `friends`
- `review_count` crudo del usuario
- `average_stars` crudo del usuario

La razon es evitar leakage o una dependencia excesiva de agregados poco fiables respecto a train.

#### Como interpretarlo

- Este bloque no representa gustos sobre items.
- Representa antiguedad, estatus y actividad historica del usuario en la plataforma.
- Conviene tratarlo como senal complementaria, no como sustituto del perfil de gustos.

#### Como manejarlo

- Si quieres una version content-based estricta, puedes desactivarlo con `--no-metadata`.
- Si quieres un regresor final de rating, suele tener sentido comparar con y sin metadata.
- Como esta separado en su propia matriz, es facil escalarlo o regularizarlo aparte.

### 3.3 Bloque profundo de competicion

La corrida de competicion usa un encoder profundo de usuario que ya existe en codigo.

Puntos clave:

- la entrada base del negocio es `business_full_features`
- el runner de competicion usa `business_view="full"` tanto para la rama manual como para la rama profunda
- el encoder profundo usa validacion temporal interna y exporta embeddings densos por `user_id`
- el resumen profundo distingue `history`, `metadata_only` y `default_only`
- el resumen profundo ahora expone `user_feature_source`, `business_feature_source`, `temporal_validation_protocol` y `export_history_source`

Interpretacion importante:

- `best_val_mae` en `user_deep_summary.json` es la validacion interna del entrenamiento
- la tabla de utilidad del reporte es un diagnostico post-export, no una comparacion final de produccion
- si un embed se calcula sobre un snapshot ya entrenado y luego se mide sobre un split temporal del mismo train, la lectura debe ser cauta porque no es una baseline leak-free completa

## 4. Convencion de nombres

Las features siguen una convencion muy util para filtrar columnas:

- negocio:
  - `geo__...`
  - `category__...`
  - `attribute__...`
  - `hours__...`
  - `prior__...`
- usuario:
  - `profile__...`
  - `metadata__...`

Esto te permite:

- recuperar todas las columnas de un bloque por prefijo
- hacer ablations faciles
- construir scorers distintos por familia de senal

Ejemplos:

- todo lo geografico del negocio: prefijo `geo__`
- todas las categorias del perfil de usuario: prefijo `profile__category__`
- toda la metadata del usuario: prefijo `metadata__`

## 5. Artefactos que debes mirar para manejar las features

### Negocio

Cuando ejecutes:

```powershell
python .\content-based\build_business_representation.py --save-dir .\content-based\artifacts\business_repr_v1
```

obtendras:

- `business_content_features.npz`
- `business_prior_features.npz`
- `business_full_features.npz`
- `business_feature_names.json`
- `feature_metadata.csv`
- `business_block_summary.csv`
- `clean_business_table.parquet`

Los mas importantes para inspeccion son:

- `business_feature_names.json`: orden exacto de columnas
- `feature_metadata.csv`: bloque, fuente y regla por feature
- `business_block_summary.csv`: rangos de indices por bloque
- `clean_business_table.parquet`: version legible y limpia de las transformaciones

### Usuario

Cuando ejecutes:

```powershell
python .\content-based\build_user_representation.py --save-dir .\content-based\artifacts\user_repr_v1
```

obtendras:

- `user_profile_features.npz`
- `user_metadata_features.npz`
- `user_full_features.npz`
- `user_feature_names.json`
- `user_feature_metadata.csv`
- `clean_user_table.parquet`

Los mas utiles para depurar son:

- `user_feature_names.json`
- `user_feature_metadata.csv`
- `clean_user_table.parquet`
- `user_profile_summary.json`, que ahora incluye `business_view`, `business_blocks` y `profile_source`

## 6. Como cargar y manipular las matrices

### Opcion A: usar los builders en memoria

```python
from pathlib import Path
import sys

sys.path.append("content-based")

from utils.io import load_businesses, load_train_reviews, load_users
from utils.business_representation import (
    BusinessRepresentationBuilder,
    BusinessRepresentationConfig,
)
from utils.user_representation import (
    UserRepresentationBuilder,
    UserRepresentationConfig,
)

data_dir = Path("content-based/data")

business_bundle = BusinessRepresentationBuilder(
    BusinessRepresentationConfig()
).fit_transform(
    load_businesses(data_dir),
    load_train_reviews(data_dir),
)

user_bundle = UserRepresentationBuilder(
    UserRepresentationConfig(
        aggregation_mode="centered",
        business_view="content",
        include_metadata=True,
    )
).fit_transform(
    train_reviews=load_train_reviews(data_dir),
    business_bundle=business_bundle,
    users_df=load_users(data_dir),
)

X_business = business_bundle.get_matrix(view="content")
X_user = user_bundle.get_matrix(view="profile")
```

### Opcion B: cargar artefactos guardados

```python
import json
import pandas as pd
from scipy import sparse

base = "content-based/artifacts/business_repr_v1_smoke"

X_business = sparse.load_npz(f"{base}/business_full_features.npz")
feature_names = json.load(open(f"{base}/business_feature_names.json", encoding="utf-8"))
feature_metadata = pd.read_csv(f"{base}/feature_metadata.csv")
block_summary = pd.read_csv(f"{base}/business_block_summary.csv")
```

## 7. Como seleccionar solo parte de las features

### Seleccionar vistas ya separadas

En negocio:

- `view="content"`
- `view="prior"`
- `view="full"`

En usuario:

- `view="profile"`
- `view="metadata"`
- `view="full"`

### Seleccionar bloques concretos

En negocio:

```python
X_categories = business_bundle.get_matrix(blocks=["categories"])
X_geo_hours = business_bundle.get_matrix(blocks=["geo", "hours"])
```

En usuario:

```python
X_user_profile_only = user_bundle.get_matrix(view="profile")
X_user_metadata_only = user_bundle.get_matrix(view="metadata")
```

Y para construir perfiles de usuario usando solo ciertos bloques de negocio:

```python
user_config = UserRepresentationConfig(
    aggregation_mode="centered",
    business_view="full",
    business_blocks=["categories", "attributes"],
)
```

Importante:

- si pasas `business_blocks`, la seleccion se hace sobre `full_matrix`
- por tanto puedes mezclar, por ejemplo, `categories` con `priors`
- pero debes hacerlo de forma consciente porque ya no seria un perfil puramente de contenido

## 8. Recomendaciones practicas para trabajar con estas features

### Si quieres recomendacion content-based pura

Usa:

- negocio: `content_matrix`
- usuario: `profile_matrix`
- configuracion recomendada: `aggregation_mode="centered"` y `business_view="content"`

Porque:

- mantienes al usuario y al negocio en el mismo espacio semantico
- evitas meter popularidad o calidad agregada como si fueran contenido

### Si quieres predecir rating con mejor MAE

Compara varias opciones:

- `user profile(content) + business content`
- `user profile(full) + business full`
- `user profile(content) + business full`
- anadir o no `user metadata`

La opcion mas predictiva no tiene por que ser la mas interpretable.

### Si quieres analizar aporte de bloques

Haz ablations como:

- solo `categories`
- solo `attributes`
- `categories + attributes`
- `geo + hours`
- solo `priors`

Eso te dira si la senal viene del contenido semantico, del contexto geografico o de la popularidad observada.

### Si quieres depurar por que una feature existe o no existe

Consulta en este orden:

1. `clean_business_table.parquet` o `clean_user_table.parquet`
2. `feature_metadata.csv` o `user_feature_metadata.csv`
3. `business_block_summary.csv`
4. la configuracion usada en el `*_summary.json`

## 9. Riesgos y precauciones

### Leakage

En este modulo ya hay una decision importante:

- no usar directamente `stars` ni `review_count` crudos de `negocios.csv`
- no usar directamente `average_stars` ni `review_count` crudos de `usuarios.csv`

Cuando anadas nuevas features, preguntate siempre:

- esto esta disponible antes de predecir
- o resume interacciones futuras respecto a mi split de train

### Reportes y metricas

El `embedding_quality_report` mezcla cobertura, salud numerica, coherencia semantica y utilidad.
No lo leas como una baseline final de produccion sin mirar el origen de cada numero.

Advertencias practicas:

- la cobertura de negocio puede subir despues de imputar `state`, `city`, `latitude` y `longitude`
- la utilidad post-export usa un scorer ligero sobre artefactos ya exportados, asi que es diagnostica y no sustituye a una evaluacion leak-free completa
- `MAE temporal oficial deep` es la mejor validacion interna del entrenamiento profundo, no el mismo objeto que la tabla de utilidad

### Cold start

Tu dataset tiene mucho `new_user_known_item`.
Por eso:

- las features de negocio son criticas
- el bloque `metadata` de usuario puede ayudar algo
- el perfil de usuario tiene limitaciones fuertes cuando solo hay una review

### Interpretabilidad

Las features mas interpretables son:

- `categories`
- gran parte de `attributes`
- `hours`

Las menos interpretables son:

- `geo__city_hash__*`
- cualquier combinacion que mezcle `profile` con `priors`

## 10. Resumen corto para tomar decisiones

- Si quieres gusto puro del usuario: usa `profile` con `business_view="content"`.
- Si quieres describir items: usa `content_matrix`.
- Si quieres senal adicional de popularidad o calidad observada: usa `priors`.
- Si quieres una version mas segura contra leakage: mantente cerca de `content` y `metadata` segura.
- Si quieres experimentar rapido: usa `business_blocks` para hacer ablations.

## 11. Referencias internas

- [Content-Based README](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/README.md)
- [Project Status](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/project-status.md)
- [Embedding Quality Report Guide](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/embedding_quality_report_guide.md)
- [Business builder](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)
- [User builder](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)
- [Competition builder](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_competition_embeddings.py)
