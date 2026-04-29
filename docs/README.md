# Documentation

## Current State

The content-based module now extends beyond the deep export and frozen regressor. The current state also includes the router `lgbm_raw_router_prefix_deep_v1`, which is the official competition reference today.

- Purpose: master index and canonical entry point for the documentation
- Document type: `current`
- Last updated: `2026-04-11`

## How to Navigate

The canonical documentation lives in `docs/`. The repository and module `README.md` files are short entry hubs. Reference information, status, architecture, flows, proposals, and experiments are maintained here to avoid duplication.

## Canonical Structure

- [Documentation Standards](STANDARDS.md)
- [Repository Map](overview/repository-map.md)
- [Current State](status/current-state.md)

## Architecture

- [Content-Based Current Architecture](architecture/content-based-current.md)
- [Decision Log](architecture/decision-log.md)

## Flows

- [Content-Based Pipeline](flows/content-based-pipeline.md)
- [Collaborative Filtering Workflow](flows/collaborative-filtering-workflow.md)

## Training and Evaluation

- [Content-Based Deep User](training/content-based-deep-user.md)
- [Content-Based Frozen Regressor](training/content-based-frozen-regressor.md)
- [Content-Based LGBM Raw Router](training/content-based-lgbm-raw-router.md)
- [Content-Based Two-Tower Router](training/content-based-known-user-two-tower-router.md)

## Reference

- [Content-Based Artifacts](reference/content-based-artifacts.md)
- [Data Assets](reference/data-assets.md)
- [Legacy Model Artifacts](reference/model-artifacts.md)
- [Notebook Inventory](reference/notebooks.md)
- [Collaborative Filtering Models](reference/collaborative-filtering-models.md)
- [Collaborative Filtering Metrics](reference/collaborative-filtering-metrics.md)
- [Collaborative Filtering Ensembles](reference/collaborative-filtering-ensemble.md)
- [Collaborative Filtering Utils](reference/collaborative-filtering-utils.md)

## Experiments

- [Official Run and Snapshot Registry](experiments/registry.md)
- [Content-Based Deep User Log](experiments/content-based-deep-user-log.md)
- [Raw Router Prefix Deep 2026-04-11](experiments/raw-router-prefix-deep-2026-04-11.md)
- [Two-Tower Router 2026-04-12](experiments/two-tower-router-2026-04-12.md)

## Proposals

- [Content-Based Interaction-First](proposals/content-based-interaction-first.md)
- [Content-Based Next Ideas](proposals/content-based-next-ideas.md)

## Document Type Convention

- `current`: describes what exists today and is considered active
- `reference`: describes contracts, inventories, assets, or APIs
- `experiment`: records iterations, results, or run recommendations
- `proposal`: documents ideas or changes not yet implemented

## Main Rule

If a piece of data can only be maintained in one place without risk of inconsistency, that place must be `docs/` and not a `README.md`.
