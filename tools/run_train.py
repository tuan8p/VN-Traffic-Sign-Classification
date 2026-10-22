from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import logging
from vn_tsc.config.resolve import resolve_config
from vn_tsc.runtime.run_dir import make_run_dir
from vn_tsc.runtime.wandb_gate import build_base_run_name
from vn_tsc.utils.seed import set_seed

def main(argv=None) -> None:
    is_main = int(os.environ.get("RANK", 0)) == 0
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    p = argparse.ArgumentParser(description="Train one pipeline")
    p.add_argument("--pipeline", required=True, choices=["svm", "boosting", "dl"])
    p.add_argument("--config", default=None)
    p.add_argument("--shared", default="configs/shared.yaml")
    p.add_argument("--runtime", default="configs/runtime/local.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-dir", default=None, help="Explicit path to output directory")
    p.add_argument("--data-root", default=None, help="Override data.processed_root")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    p.add_argument("--no-eval-test", action="store_true", help="Skip auto-evaluating on test set after training")
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.data_root:
        overrides.setdefault("data", {})["processed_root"] = args.data_root
    if args.no_wandb:
        overrides.setdefault("train", {})["require_wandb"] = False
    if args.no_eval_test:
        overrides.setdefault("eval", {})["auto_eval_test"] = False

    pipe_yaml = args.config or f"configs/pipelines/{args.pipeline}.yaml"
    cfg = resolve_config(
        pipeline_yaml=pipe_yaml,
        shared_yaml=args.shared,
        runtime_yaml=args.runtime,
        overrides=overrides if overrides else None,
    )
    set_seed(int(cfg.get("seed", 42)))

    base_name = build_base_run_name(args.pipeline, cfg, user_name=args.run_name)
    cfg.setdefault("train", {})["run_name"] = base_name

    if args.run_dir:
        run_dir = Path(args.run_dir)
        for sub in ("logs", "figures", "checkpoints"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg.get("outputs", {}).get("root", "outputs"), args.pipeline, base_name)

    if args.pipeline == "svm":
        from vn_tsc.pipelines.svm.train import SVMPipeline as P
    elif args.pipeline == "boosting":
        from vn_tsc.pipelines.boosting.train import BoostingPipeline as P
    else:
        from vn_tsc.pipelines.dl.train import DLPipeline as P

    metrics = P(cfg, run_dir).fit()
    if int(os.environ.get("RANK", 0)) == 0:
        print("run_dir:", run_dir)
        print("metrics:", metrics)

if __name__ == "__main__":
    main()

