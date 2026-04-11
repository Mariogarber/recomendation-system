# Documentacion Del Repositorio

## Estado Vigente

La rama content-based ya no se describe solo por el deep export y el frozen regressor. El estado actual tambien incluye el router `lgbm_raw_router_prefix_deep_v1`, que es la referencia oficial hoy para la competencia.

- Proposito: indice maestro y punto de entrada canónico de la documentacion.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-11`

## Como navegar esta documentacion

La documentacion canónica del repositorio vive en `docs/`.

Los `README.md` del repositorio y de cada modulo son hubs de entrada cortos. La informacion de referencia, estado, arquitectura, flujos, propuestas y experimentos se mantiene aqui para evitar duplicacion.

## Estructura canónica

- [Estándares Documentales](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/STANDARDS.md)
- [Mapa Del Repositorio](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/overview/repository-map.md)
- [Estado Actual](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/status/current-state.md)

## Arquitectura

- [Content-Based Actual](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/content-based-current.md)
- [Decision Log](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/decision-log.md)

## Flujos

- [Pipeline Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/flows/content-based-pipeline.md)
- [Workflow Collaborative Filtering](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/flows/collaborative-filtering-workflow.md)

## Training Y Evaluacion

- [Deep User De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-deep-user.md)
- [Frozen Regressor De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-frozen-regressor.md)
- [LGBM Raw Router De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-lgbm-raw-router.md)

## Referencia

- [Artefactos De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/content-based-artifacts.md)
- [Datasets Y Activos De Datos](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/data-assets.md)
- [Artefactos De Modelos Legacy](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/model-artifacts.md)
- [Inventario De Notebooks](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/notebooks.md)
- [Modelos De Collaborative Filtering](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/collaborative-filtering-models.md)
- [Metricas De Collaborative Filtering](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/collaborative-filtering-metrics.md)
- [Ensembles De Collaborative Filtering](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/collaborative-filtering-ensemble.md)
- [Utils De Collaborative Filtering](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/collaborative-filtering-utils.md)

## Experimentos

- [Registro Oficial De Runs Y Snapshots](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/registry.md)
- [Log Deep User De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/content-based-deep-user-log.md)
- [Raw Router Prefix Deep 2026-04-11](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/raw-router-prefix-deep-2026-04-11.md)

## Propuestas

- [Content-Based Interaction-First](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/proposals/content-based-interaction-first.md)
- [Siguientes Ideas De Content-Based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/proposals/content-based-next-ideas.md)

## Convencion De Tipos Documentales

- `current`: describe lo que existe hoy y se considera vigente.
- `reference`: describe contratos, inventarios, activos o APIs.
- `experiment`: registra iteraciones, resultados o recomendaciones de runs.
- `proposal`: documenta ideas o cambios aun no implementados.

## Regla Principal

Si un dato solo puede mantenerse en un sitio sin riesgo de incoherencia, ese sitio debe ser `docs/` y no un `README.md`.
