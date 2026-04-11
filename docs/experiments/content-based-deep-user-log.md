# Log Deep User De Content-Based

- Proposito: resumir las iteraciones historicas relevantes de la familia deep de `content-based`.
- Tipo documental: `experiment`
- Ultima actualizacion: `2026-04-11`

## Iteraciones Registradas

- `competition_embeddings_v1`
  - primera familia exportable documentada
- `competition_embeddings_v2_smoke`
  - smoke test de pipeline
- `competition_embeddings_v3_iter01`
  - iteracion historica intermedia
- `competition_embeddings_v3_iter02`
  - iteracion historica intermedia
- `competition_embeddings_v3_iter03`
  - snapshot oficial actual para export de embeddings
- `competition_embeddings_v3_iter04`
  - snapshot candidato usado como referencia de comparacion downstream
- `lgbm_raw_router_prefix_deep_v1`
  - router actual que reutiliza el bundle deep oficial como fuente de embeddings de negocio para la rama `known_prefix_deep_model`

## Evolucion De Enfoques

- manual
  - representacion explicita de negocio y perfil agregado de usuario
- deep export
  - encoder profundo con export de embeddings de usuario y negocio
- frozen downstream / leak audit
  - uso de embeddings exportados para diagnostico y scorers congelados
- raw_core
  - baseline tabular fuerte para usuarios con historial
- router cold archetypes
  - cold start resuelto con arquetipos metadata-only
- router prefix-deep
  - known users intermedios refinados con embeddings deep exportados y routing por banda

## Regla

Este documento resume historia. La declaracion oficial de estado vive en:

- [Registro Oficial De Runs Y Snapshots](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/registry.md)
