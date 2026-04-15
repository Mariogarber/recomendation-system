import json

runs = {
    'lgbm_train_stars_v1 (baseline)': 'artifacts/lgbm_train_stars_v1/validation_summary.json',
    'lgbm_router_v5      (64 arc, no biz stats)': 'artifacts/lgbm_router_v5/validation_summary.json',
}
for name, p in runs.items():
    with open(p) as f:
        s = json.load(f)
    bands = {b['history_band']: b['mae'] for b in s['band_metrics_router']}
    key = 'router_validation_mae_rounded'
    print(name)
    for k in ['0','1','2-5','6-20','>20']:
        print(f'  band {k}: {bands[k]:.4f}')
    print(f'  global: {s[key]:.4f}')
    print()

print('=== ESTIMATED LB IMPROVEMENT ===')
with open('artifacts/lgbm_train_stars_v1/validation_summary.json') as f: v1 = json.load(f)
with open('artifacts/lgbm_router_v5/validation_summary.json') as f: v5 = json.load(f)
b1 = {b['history_band']: b['mae'] for b in v1['band_metrics_router']}
b5 = {b['history_band']: b['mae'] for b in v5['band_metrics_router']}
cold_delta = b5['0'] - b1['0']
lb_delta = 0.41 * cold_delta
print(f'  cold band MAE delta: {cold_delta:+.4f}')
print(f'  41pct cold * delta:  {lb_delta:+.4f}')
print(f'  current LB:          0.6528')
print(f'  estimated new LB:    {0.6528 + lb_delta:.4f}')
