# Decision Log De Arquitectura

- Proposito: registrar decisiones arquitectonicas relevantes, su estado y su impacto sobre codigo, flows y artefactos.
- Tipo documental: `current`
- Ultima actualizacion: `2026-04-10`

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
