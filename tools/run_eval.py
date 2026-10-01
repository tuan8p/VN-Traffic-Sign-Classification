from __future__ import annotations

import argparse
from pathlib import Path

from vn_tsc.utils.io import load_yaml


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True, choices=["svm", "boosting", "dl"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--processed-root", default=None)
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    if args.pipeline == "svm":
        from vn_tsc.pipelines.svm.train import SVMPipeline

        if cfg.get("pipeline") != "svm":
            raise ValueError("The saved run is not an SVM pipeline")
        if args.processed_root:
            cfg.setdefault("data_source", {})["processed_root"] = args.processed_root
        cfg.setdefault("eval", {})["split"] = args.split
        metrics = SVMPipeline(cfg, run_dir).evaluate()
        print("split:", args.split)
        print("metrics:", metrics)
        return
    if args.pipeline == "boosting":
        from vn_tsc.pipelines.boosting.train import BoostingPipeline

        if cfg.get("pipeline") != "boosting":
            raise ValueError("The saved run is not a boosting pipeline")
        if args.processed_root:
            cfg.setdefault("data", {})["processed_root"] = args.processed_root
        cfg.setdefault("eval", {})["split"] = args.split
        metrics = BoostingPipeline(cfg, run_dir).evaluate()
        print("split:", args.split)
        print("metrics:", metrics)
        return
    raise NotImplementedError(f"TODO(team-{args.pipeline}): eval from {run_dir}")

if __name__ == "__main__":
    main()
