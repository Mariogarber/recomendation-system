# Decision Log De Arquitectura

- Proposito: registrar decisiones arquitectonicas relevantes, su estado y su impacto sobre codigo, flows y artefactos.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-11`

## Como Leer Este Log

Estados usados:

- `proposed`
- `accepted`
- `implemented`
- `superseded`
- `rejected`

## ADR-001: `docs/` pasa a ser la casa canónica de documentacion

- Estado: `implemented`
- Fecha: `2026-04-10`
- Contexto:
  - la documentacion estaba fragmentada entre `README.md`, `content-based/README.md`, `docs/` y documentos largos de modulo
- Decision:
  - la documentacion transversal pasa a vivir en `docs/`
  - los `README.md` quedan como hubs cortos
- Consecuencias:
  - se reducen duplicaciones
  - los documentos antiguos pasan a ser stubs de transicion

## ADR-002: La arquitectura deep vigente mantiene una `business_tower` compartida

- Estado: `implemented`
- Fecha: `2026-04-10`
- Contexto:
  - la implementacion actual del deep encoder reutiliza la misma torre para historial y candidato
- Decision:
  - documentar explicitamente esa arquitectura como estado actual

## ADR-003: El frozen regressor se considera etapa oficial downstream del pipeline

- Estado: `implemented`
- Fecha: `2026-04-10`
- Contexto:
  - el script downstream existia pero no estaba tratado como etapa oficial en la narrativa principal
- Decision:
  - incorporar `train_frozen_embedding_regressor.py` y su modelo asociado a los flows y training del modulo

## ADR-004: Separar encoding de historial y encoding del negocio candidato

- Estado: `proposed`
- Fecha: `2026-04-10`
- Contexto:
  - hoy el `user_embedding` no recibe directamente el candidato, pero historial y candidato siguen compartiendo la misma `business_tower`
  - eso deja un acoplamiento indirecto entre ambos roles
- Decision propuesta:
  - introducir una torre para negocios del historial y otra para el negocio candidato
  - mantener `user_embedding` dependiente solo de historial agregado y metadata
- Alternativas consideradas:
  - mantener una sola torre y solo clarificar la separacion conceptual
  - mover el sistema a una arquitectura `interaction-first`
- Impacto esperado:
  - separar mejor preferencias del usuario y representacion del item a predecir
  - obligar a regenerar embeddings deep y checkpoints
- Seguimiento requerido si se implementa:
  - actualizar arquitectura, flows, training, contratos de artefactos y registro de experimentos

## ADR-005: Router prefix-deep para usuarios conocidos de historia corta y media

- Estado: `implemented`
- Fecha: `2026-04-11`
- Contexto:
  - el router `raw_core` + arquetipos mejoro la captura de cold start, pero seguia habiendo margen en usuarios conocidos intermedios
  - los intentos deep anteriores no daban una mejora estable como scorer global, pero si dejaban un bundle deep utilizable como fuente de embeddings
  - el run nuevo mostro que el candidato prefix-deep solo merece activarse en `6-20` con un margen minimo de mejora
- Decision:
  - mantener `cold_model` para `history_band = 0`
  - mantener `known_model` como fallback para usuarios conocidos que no entren en la activacion prefix-deep
  - activar `known_prefix_deep_model` solo para `history_band = 6-20`
  - considerar `lgbm_raw_router_prefix_deep_v1` como snapshot oficial del router
- Evidencia:
  - `validation_mae_rounded = 0.6265079379`
  - baseline previo `lgbm_raw_router_v1 = 0.6268747449`
  - `6-20` mejora a `0.6845791340`
  - `1` empeora a `0.7012391090`
  - `2-5` mejora a `0.7327732444` pero no supera el margen de activacion
- Consecuencias:
  - el router actual queda como combinacion de `raw_core`, arquetipos y prefix-deep
  - la banda `2-5` sigue siendo la mejor candidata para la siguiente iteracion de perfilado de usuario
  - los docs de estado, training y artefactos deben tratar `lgbm_raw_router_v1` como referencia historica, no como snapshot oficial vigente
