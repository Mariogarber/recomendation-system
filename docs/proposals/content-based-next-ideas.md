# Siguientes Ideas De Content-Based

- Proposito: mantener un backlog corto de ideas y direcciones futuras no implementadas.
- Tipo documental: `proposal`
- Ultima actualizacion: `2026-04-14`

## Experimento En Curso

- `lgbm_train_stars_v1`: reemplaza `user_average_stars` (Yelp all-time, posible leakage) por media calculada exclusivamente desde `train_reviews`; si el leaderboard mejora, aplicar el mismo fix al modelo deep (`known_user_deep_e2e`, `KNOWN_USER_NUMERIC_FEATURE_COLUMNS`)

## Ideas Activas

- separar encoder de historial y encoder del negocio candidato
- revisar si conviene una doble torre o una variante hibrida
- formalizar una suite leak-safe de baselines content-based
- definir una politica explicita de cold start
- comparar mejor manual user vs deep user vs downstream frozen regressor
- reinyectar las nuevas features tabulares del experimento `feature-first` dentro de la linea `known_user_deep_router_v2_eval_v3`
- probar un router tabular mas conservador para `2-5`:
  - mantener `known_prefix` solo en `6-20`
  - desactivar `transition_blend` en `2-5` y medir si el global aguanta
- si se reabre la especializacion corta, hacerlo con foco explicito en `2-3`, no en toda la banda `2-5`

## Contexto Nuevo Ya Asumido

- el diagnostico `known_user_short_band_diagnostic_v1` confirmo que la banda corta no es homogenea y que `2-3` sigue siendo el mayor problema
- la tanda `lgbm_feature_first_short_router_v1_gpu` mejoro ligeramente el MAE global tabular, pero empeoro todos los segmentos `2`, `3`, `4`, `5`, `2-3` y `4-5` frente a `known_user_deep_router_v2_eval_v3`
- por tanto la siguiente iteracion no deberia reemplazar a `v3` por un router tabular corto, sino reaprovechar esas features donde no destruyan la ventaja del deep

## Regla

- si una idea madura y cambia arquitectura, debe entrar tambien en `decision-log.md`
- si una idea se implementa, deja de vivir aqui y pasa a `current-state.md`, `flows/` y `architecture/`
