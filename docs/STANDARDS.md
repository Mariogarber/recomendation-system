# Estandares Documentales

- Proposito: guiar a futuros agentes y contribuidores para ampliar la documentacion sin romper la arquitectura documental.
- Tipo documental: `reference`
- Ultima actualizacion: `2026-04-10`

## Principios

- `docs/` es la unica casa canónica de documentacion transversal del repositorio.
- Los `README.md` son hubs de entrada, no fuentes completas de verdad.
- Cada documento debe tener un unico objetivo claro.
- No se duplica contenido estable si puede enlazarse al documento canónico.
- Todo cambio relevante en scripts, artefactos o arquitectura debe reflejarse en los documentos de `flow`, `reference` y `status` correspondientes.

## Header Obligatorio

Todo documento nuevo en `docs/` debe comenzar con estas tres lineas o equivalentes:

- `Proposito`
- `Tipo documental`
- `Ultima actualizacion`

Tipos permitidos:

- `current`
- `reference`
- `experiment`
- `proposal`

## Idioma

- Idioma por defecto: espanol.
- Se permite ingles solo si un documento ya es legacy o si hay una razon fuerte de interoperabilidad externa.
- No mezclar idiomas dentro del mismo documento salvo nombres propios, comandos o identificadores tecnicos.

## Donde debe vivir cada cosa

- `docs/overview/`: mapas del repo y orientacion general
- `docs/status/`: estado actual y roadmap vigente
- `docs/architecture/`: arquitectura implementada y log de decisiones
- `docs/flows/`: pasos operativos end-to-end y runbooks
- `docs/training/`: protocolos de entrenamiento, evaluacion y seleccion
- `docs/reference/`: contratos, inventarios, datasets, artefactos, APIs y notebooks
- `docs/experiments/`: logs de iteraciones, snapshots recomendados y resultados historicos
- `docs/proposals/`: RFCs, ideas futuras y redisenos aun no implementados

## Reglas De Separacion

- No mezclar arquitectura implementada con ideas futuras en el mismo documento.
- No mezclar protocolos estables con logs experimentales.
- No declarar snapshots oficiales en varios sitios.
- No poner tablas largas de artefactos en un `README.md` si ya existe un documento de referencia.

## Single Source Of Truth

Usa estas fuentes de verdad:

- estado actual del repo: [current-state.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/status/current-state.md)
- arquitectura content-based vigente: [content-based-current.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/content-based-current.md)
- decisiones arquitectonicas: [decision-log.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/decision-log.md)
- snapshots y runs recomendados: [registry.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/experiments/registry.md)
- contratos de artefactos content-based: [content-based-artifacts.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/content-based-artifacts.md)

## Cuando Crear O Actualizar Cada Documento

- Crear o actualizar `docs/proposals/` cuando aparece una idea nueva que cambia flujo, arquitectura o artefactos y aun no esta implementada.
- Actualizar `docs/architecture/decision-log.md` cuando una decision se propone, se acepta, se rechaza o queda supersedida.
- Actualizar `docs/architecture/content-based-current.md` cuando la arquitectura implementada cambia de verdad.
- Actualizar `docs/flows/` cuando cambia el orden operativo de un pipeline o se anade una etapa ejecutable.
- Actualizar `docs/training/` cuando cambia el split, las metricas, la regla de seleccion o el comando recomendado.
- Actualizar `docs/reference/` cuando cambia un contrato de archivos, una API, un inventario o el significado de un artefacto.
- Actualizar `docs/status/current-state.md` cuando el estado del repo o el snapshot recomendado cambia.
- Actualizar `docs/experiments/registry.md` cuando un run pasa a ser `official`, `candidate` o `deprecated`.

## Reglas Para READMEs

- `README.md` del repo: onboarding corto, mapa rapido y enlace a `docs/README.md`.
- `README.md` de modulo: contexto del modulo, scripts principales y enlace a la documentacion canónica.

Un `README.md` no debe convertirse en decision log, contrato de artefactos, log de experimentos ni protocolo largo de training.

## Politica De Documentos Legacy

- Si un documento antiguo deja de ser canónico, no se borra de inmediato si hay enlaces entrantes probables.
- Se convierte en un stub corto con aviso de legacy y enlace al documento canónico nuevo.
- El contenido duplicado se elimina del legacy para evitar divergencia.

## Checklist Antes De Crear Un Documento Nuevo

1. comprobar si el contenido ya pertenece a un documento existente
2. comprobar si en realidad basta con actualizar `flow`, `reference`, `status` o `proposal`
3. comprobar si el nombre deja claro el tipo de contenido
4. comprobar si el contenido sera estable o es solo un log experimental

## Checklist Antes De Fusionar Cambios Documentales

- El documento tiene `Proposito`, `Tipo documental` y `Ultima actualizacion`.
- No duplica una fuente de verdad ya existente.
- Los enlaces internos apuntan a la nueva estructura canónica.
- Si el cambio afecta scripts o artefactos, tambien se actualizaron `flow` y `reference`.
- Si el cambio altera recomendaciones, tambien se actualizo `registry.md`.
- Si el cambio altera arquitectura, tambien se actualizo `decision-log.md`.

## Anti-Patrones A Evitar

- crear un RFC que describe algo ya implementado como si aun fuera solo propuesta
- meter resultados experimentales dentro de un documento de arquitectura actual
- declarar dos snapshots distintos como “el recomendado” en documentos diferentes
- copiar tablas completas entre `README`, `status` y `reference`
- dejar documentos con versiones o paths historicos como si fueran la configuracion vigente
