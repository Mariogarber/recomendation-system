# Workflow De Collaborative Filtering

- Proposito: resumir el flujo operativo vigente de la rama legacy `colaborative-filtering`.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-10`

## Objetivo

La rama `colaborative-filtering` agrupa modelos de prediccion de rating basados en interacciones usuario-item, utilidades de evaluacion y estrategias de ensemble.

## Flujo Operativo Tipico

1. cargar `data/train.csv` y `data/test.csv`
2. elegir familia de modelo
3. entrenar sobre `train.csv`
4. evaluar con MAE o RMSE
5. opcionalmente combinar modelos con ensemble
6. generar predicciones para `test.csv`
7. guardar CSV de salida

## Componentes De Trabajo

- modelos:
  - [`colaborative-filtering/model`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/model)
- metricas:
  - [`colaborative-filtering/metric`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/metric)
- ensembles:
  - [`colaborative-filtering/ensemble`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/ensemble)
- utilidades:
  - [`colaborative-filtering/utils`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/utils)

## Relacion Con Notebooks

Notebooks principales:

- [`analysis.ipynb`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/analysis.ipynb)
- [`hiperparameter.ipynb`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/hiperparameter.ipynb)
- [`knn.ipynb`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/knn.ipynb)
- [`notebook_svd_knn.ipynb`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/notebook_svd_knn.ipynb)
- [`recomendation-system.ipynb`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering/recomendation-system.ipynb)
