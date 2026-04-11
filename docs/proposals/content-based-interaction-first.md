# Propuesta: Content-Based Interaction-First

- Proposito: consolidar la propuesta de arquitectura `interaction-first` en un unico documento de propuestas.
- Tipo documental: `proposal`
- Ultima actualizacion: `2026-04-10`

## Idea Central

Cambiar el orden del procesamiento para que:

- primero se agregue el historial del usuario
- despues ese contexto condicione el procesamiento del negocio candidato

En lugar de:

- `business tower` compartida
- `user embedding`
- `scorer`

la propuesta pasa a:

- `history encoder`
- `history aggregator`
- `metadata encoder`
- `interaction tower`
- `rating head`

## Motivacion

- evitar que el candidato domine demasiado la prediccion
- introducir el contexto historico antes en la ruta principal
- acercar el modelo a un matching contextual usuario-candidato

## Tradeoffs

- mejora potencial en prediccion contextual
- peor reutilizacion de embeddings de negocio puros
- menor modularidad que la arquitectura actual

## Estado

- propuesta documentada
- no implementada
- no sustituye la arquitectura vigente descrita en `content-based-current.md`
