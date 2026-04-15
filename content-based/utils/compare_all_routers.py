import json

paths = {
    'lgbm_train_stars_v1 (64 arc, no biz stats, all bands)': 'artifacts/lgbm_train_stars_v1/validation_summary.json',
    'lgbm_router_v5      (64 arc, no biz stats, all bands)': 'artifacts/lgbm_router_v5/validation_summary.json',
    'lgbm_router_v4      (32 arc, biz stats, all bands)':    'artifacts/lgbm_router_v4/validation_summary.json',
    'lgbm_router_v3      (128 arc, biz stats, all bands)':   'artifacts/lgbm_router_v3/validation_summary.json',
    'lgbm_router_v2      (128 arc, biz stats, band 1+2-5)':  'artifacts/lgbm_router_v2/validation_summary.json',
}

for label, p in paths.items():
    with open(p) as f:
        s = json.load(f)
    cm = s['cold_model']
    bands = {b['history_band']: b['mae'] for b in s['band_metrics_router']}
    n_arc = s['spec_config']['n_user_archetypes']
    total_feat = cm['feature_summary']['total']
    train_rows = cm['train_rows']
    biz_feat = cm['feature_summary']['business']
    global_mae = s['router_validation_mae_rounded']
    cold_mae = bands['0']
    print(label)
    print(f'  cold_train_rows={train_rows}  archetypes={n_arc}  cold_features={total_feat} (biz={biz_feat})')
    print(f'  band 0 MAE={cold_mae:.4f}  global MAE={global_mae:.4f}')
    print()
