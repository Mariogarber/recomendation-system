# Content-Based LGBM Feature-First Short Router

- Date: `2026-04-13`
- Status: `implemented, evaluated, not promoted`
- Scope: feature-first ablation over the transition-blend LightGBM router for short known-user history, trained with `uv` and GPU-enabled LightGBM

## Goal

Probar una linea `feature-first` antes de seguir complicando la arquitectura deep en la antigua banda `2-5`.

La hipotesis de trabajo era:

- reforzar señales tabulares y de prefijo para corto historial
- mantener el deep como referencia fuerte, no como punto de partida obligatorio
- medir si una mejora de features podia cerrar parte del gap en `2-3` sin abrir otra linea arquitectonica grande

## Scripts

- training wrapper:
  - [`content-based/train_lgbm_feature_first_short_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_feature_first_short_router.py)
- transition trainer base:
  - [`content-based/train_lgbm_transition_blend_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_transition_blend_router.py)
- submission generator compatible:
  - [`content-based/predict_lgbm_transition_blend_submission.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/predict_lgbm_transition_blend_submission.py)

## Features Added In This Iteration

Raw and short-history additions:

- richer item support signals
- explicit `item_is_new`
- exact short-count flags for `2`, `3`, `4`, `5`
- support and history interactions
- short-history stability features
- prefix similarity interactions with exact history count

Main implementation points:

- [`content-based/utils/lgbm_raw_features.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/lgbm_raw_features.py)
- [`content-based/utils/lgbm_known_prefix_deep_features.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/lgbm_known_prefix_deep_features.py)

## GPU Runtime Note

The transition trainer now accepts optional GPU flags through the shared LightGBM param builder.

Relevant implementation:

- [`content-based/train_lgbm_raw_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_raw_router.py)
- [`content-based/train_lgbm_transition_blend_router.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_transition_blend_router.py)

Used runtime in the promoted experiment artifact:

- `use_gpu = true`
- `gpu_platform_id = 0`
- `gpu_device_id = 0`
- `gpu_max_bin = 255`

## Canonical Artifact

- [`content-based/artifacts/lgbm_feature_first_short_router_v1_gpu`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu)

Key files:

- [`validation_summary.json`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu/validation_summary.json)
- [`validation_predictions.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu/validation_predictions.csv)
- [`short_history_vs_v3.csv`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/lgbm_feature_first_short_router_v1_gpu/short_history_vs_v3.csv)

## Results

Against the tabular baseline `lgbm_hybrid_conservative_v1`:

- `validation_mae_rounded = 0.6254230`
- previous baseline `= 0.6265079`
- delta `= -0.0010849`

Branch-level reading:

- `cold_model` remained strong for `history_band = 0`
- `known_prefix_deep_model` was enabled only for `6-20`
- `transition_blend_model` remained active for `2-5`

Critical short-band reading:

- `2-5` route MAE `= 0.7396470`
- `known_model` on the same `2-5` rows `= 0.7357010`
- transition blend delta vs known model `= +0.0039460`

Comparison against the deep stable reference [`known_user_deep_router_v2_eval_v3`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v3):

- `2 = +0.04049`
- `3 = +0.03613`
- `4 = +0.03395`
- `5 = +0.03425`
- `2-3 = +0.03884`
- `4-5 = +0.03408`

## Decision

This run is useful as a documented ablation, but it is not a candidate submission.

Reason:

- it improves the global tabular router slightly
- it confirms the new features are usable
- but it loses clearly in the short-history zone that matters most for this line of work

Operational decision:

- do not generate or promote a submission from this artifact
- keep [`known_user_deep_router_v2_eval_v3`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/known_user_deep_router_v2_eval_v3) as the stable reference for short known-user history

## Recommended Next Step

If this line is resumed in a later session, start from one of these two paths:

- conservative tabular follow-up:
  - keep the new features
  - keep `known_prefix` for `6-20`
  - disable or redesign the `transition_blend` policy for `2-5`
- deep follow-up:
  - inject the new tabular short-history signals into the deep `v3` line instead of replacing it with the current tabular short expert

## Reproduction Command

```bash
uv run python .\content-based\train_lgbm_feature_first_short_router.py --save-root .\content-based\artifacts\lgbm_feature_first_short_router_v1_gpu --use-gpu
```
