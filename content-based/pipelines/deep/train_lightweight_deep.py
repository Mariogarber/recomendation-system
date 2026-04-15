"""Lightweight / Ultra-light deep corrector launcher.

Two config families are available via --config-family:

  v_lightweight  (~200k params, embedding_dim=32)
    business_tower: 64; scorer: 64→32; attention: 2 heads × 32 dim
    lr=1e-3, dropout=0.20, correction_scale 6-20=1.0
    Result: oscillating val_mae (±0.05); best delta band 6-20 = −0.007

  v_ultralight   (~50k params, embedding_dim=16)  [default]
    business_tower: 32; scorer: 32; attention: 2 heads × 16 dim
    lr=2e-4, batch=2048, dropout=0.30, correction_scale 6-20=0.50
    Motivation: v_lightweight showed val_mae instability from large correction
    budget (scale=1.0) + high lr. Tight scales + lower lr → stable curve.

Usage
-----
    # Ultra-light (default, recommended)
    uv run python pipelines/deep/train_lightweight_deep.py --max-runs 2

    # Lightweight (original 200k)
    uv run python pipelines/deep/train_lightweight_deep.py \\
        --config-family v_lightweight --max-runs 2

The script delegates entirely to train_known_user_deep.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "train_known_user_deep.py"
_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train lightweight deep corrector (~200k params). "
        "Passes all arguments through to train_known_user_deep.py with "
        "--config-family v_lightweight forced."
    )
    parser.add_argument(
        "--config-family",
        type=str,
        default="v_ultralight",
        choices=["v_lightweight", "v_ultralight"],
        help="v_ultralight (~50k params, stable) or v_lightweight (~200k params)",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=None,
        help="Artifact root. Defaults to artifacts/known_user_deep_<config_family>_v1",
    )
    parser.add_argument(
        "--incumbent-root",
        type=Path,
        default=_ARTIFACTS / "lgbm_raw_router_prefix_deep_v1",
    )
    parser.add_argument(
        "--business-repr-root",
        type=Path,
        default=_ARTIFACTS / "competition_embeddings_v3_iter03" / "business_repr",
    )
    parser.add_argument("--max-runs", type=int, default=2, help="1=runA only, 2=runA+runB, 3=all")
    parser.add_argument("--run-name", type=str, default=None, help="Run only this specific run by name (e.g. runC_lw_emb32_stable_lr).")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--n-user-archetypes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args, extra = parser.parse_known_args()

    save_root = args.save_root or (_ARTIFACTS / f"known_user_deep_{args.config_family}_v1")
    cmd = [
        sys.executable,
        str(_SCRIPT),
        "--config-family", args.config_family,
        "--save-root", str(save_root),
        "--incumbent-root", str(args.incumbent_root),
        "--business-repr-root", str(args.business_repr_root),
        "--max-runs", str(args.max_runs),
        "--validation-size", str(args.validation_size),
        "--max-history-len", str(args.max_history_len),
        "--n-user-archetypes", str(args.n_user_archetypes),
        "--seed", str(args.seed),
        *(["--run-name", args.run_name] if args.run_name else []),
        *extra,
    ]
    print("Launching:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
