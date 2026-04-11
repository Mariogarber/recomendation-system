# Datasets Y Activos De Datos

- Proposito: inventariar datasets, snapshots y activos de datos usados por el repositorio.
- Tipo documental: `reference`
- Ultima actualizacion: `2026-04-10`

## Dataset Legacy De Collaborative Filtering

En:

- [`data/train.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/data/train.csv)
- [`data/test.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/data/test.csv)

Otros CSVs en `data/`:

- soluciones legacy
- predicciones de ensembles
- salidas de threshold ensembles

## Dataset De Content-Based

Se asume en:

- [`content-based/data`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/data)

Tablas fuente:

- `usuarios.csv`
- `negocios.csv`
- `train_reviews.csv`
- `test_reviews.csv`

## Hechos Operativos Ya Asumidos En Content-Based

- dataset muy disperso
- cold start dominado por nuevos usuarios
- metadata de usuario y negocio con riesgo de leakage si se usa sin auditoria
- fuerte valor de la metadata estructurada de negocio

## Politica De Leakage

- no usar agregados crudos de metadata como si fueran priors limpios
- recalcular priors desde `train_reviews.csv` cuando aplique
- distinguir siempre entre metadata segura y metadata leakage-prone
