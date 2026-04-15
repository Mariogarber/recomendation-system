import json, sys

v2_path = sys.argv[1] if len(sys.argv) > 1 else 'artifacts/lgbm_router_v2/validation_summary.json'
with open(v2_path) as f:
    v2 = json.load(f)
with open('artifacts/lgbm_train_stars_v1/validation_summary.json') as f:
    v1 = json.load(f)

print('=== GLOBAL MAE ===')
print(f'  lgbm_train_stars_v1 (old):  {v1["router_validation_mae_rounded"]:.6f}')
print(f'  lgbm_router_v2      (new):  {v2["router_validation_mae_rounded"]:.6f}')
print(f'  delta:                      {v2["router_validation_mae_rounded"] - v1["router_validation_mae_rounded"]:+.6f}')

print()
print('=== BAND-LEVEL COMPARISON ===')
v1_bands = {b['history_band']: b for b in v1['band_metrics_router']}
v2_bands = {b['history_band']: b for b in v2['band_metrics_router']}
for band in ['0', '1', '2-5', '6-20', '>20']:
    m1 = v1_bands[band]['mae']
    m2 = v2_bands[band]['mae']
    n  = v1_bands[band]['n_samples']
    tag = '  <-- COLD MODEL' if band == '0' else ''
    print(f'  band {band:>4}: old={m1:.4f}  new={m2:.4f}  delta={m2-m1:+.4f}  n={n}{tag}')

print()
print('=== COLD MODEL TRAINING ROWS ===')
cm1 = v1.get('cold_model', {})
cm2 = v2.get('cold_model', {})
print(f'  v1: {cm1.get("train_rows")}  bands: {cm1.get("train_history_bands")}')
print(f'  v2: {cm2.get("train_rows")}  bands: {cm2.get("train_history_bands")}')

print()
print('=== ARCHETYPES ===')
print(f'  v1: {v1["spec_config"].get("n_user_archetypes")}')
print(f'  v2: {v2["spec_config"].get("n_user_archetypes")}')

print()
print('=== COLD FEATURE COUNTS ===')
print(f'  v1: {v1.get("cold_model", {}).get("feature_summary")}')
print(f'  v2: {v2.get("cold_model", {}).get("feature_summary")}')

print()
print('=== ESTIMATED LB IMPACT (41pct cold) ===')
cold_v1 = v1_bands['0']['mae']
cold_v2 = v2_bands['0']['mae']
lb_old = 0.6528
lb_est = lb_old + 0.41 * (cold_v2 - cold_v1)
print(f'  Cold MAE delta: {cold_v2 - cold_v1:+.4f}')
print(f'  Expected LB change: {0.41 * (cold_v2 - cold_v1):+.4f}')
print(f'  Estimated LB (new): {lb_est:.4f}')
