import json
with open('artifacts/known_user_deep_router_v2_eval_v3/known_user_deep_config.json') as f:
    cfg = json.load(f)
tc = cfg['training_config']
print('TRAINING CONFIG KEYS:', list(tc.keys()))
print()
for k, v in tc.items():
    if not isinstance(v, (list, dict)):
        print(f'  {k}: {v}')
print()
print('RUNS:')
for run in tc.get('runs', []):
    rname = run.get('run_name')
    bands = run.get('target_bands')
    lr = run.get('learning_rate')
    epochs = run.get('max_epochs')
    alpha = run.get('alpha_init')
    clamp = run.get('correction_clamp_abs')
    print(f'  name={rname}  bands={bands}  lr={lr}  epochs={epochs}  alpha={alpha}  clamp={clamp}')
