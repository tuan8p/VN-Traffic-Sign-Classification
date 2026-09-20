from __future__ import annotations
import argparse
from vn_tsc.config.resolve import resolve_config
from vn_tsc.runtime.run_dir import make_run_dir
from vn_tsc.utils.seed import set_seed

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train one pipeline (W&B required)")
    p.add_argument("--pipeline", required=True, choices=["svm", "boosting", "dl"])
    p.add_argument("--config", default=None)
    p.add_argument("--shared", default="configs/shared.yaml")
    p.add_argument("--runtime", default="configs/runtime/local.yaml")
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)
    pipe_yaml = args.config or f"configs/pipelines/{args.pipeline}.yaml"
    cfg = resolve_config(pipeline_yaml=pipe_yaml, shared_yaml=args.shared, runtime_yaml=args.runtime)
    set_seed(int(cfg.get("seed", 42)))
    run_dir = make_run_dir(cfg.get("outputs", {}).get("root", "outputs"), args.pipeline, args.run_name)
    if args.pipeline == "svm":
        from vn_tsc.pipelines.svm.train import SVMPipeline as P
    elif args.pipeline == "boosting":
        from vn_tsc.pipelines.boosting.train import BoostingPipeline as P
    else:
        from vn_tsc.pipelines.dl.train import DLPipeline as P
    metrics = P(cfg, run_dir).fit()
    print("run_dir:", run_dir)
    print("metrics:", metrics)

if __name__ == "__main__":
    main()
