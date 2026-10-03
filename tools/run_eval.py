from __future__ import annotations
import argparse
from pathlib import Path
from vn_tsc.config.resolve import resolve_config
from vn_tsc.utils.io import load_yaml


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a saved DL (or SVM/Boosting) checkpoint")
    p.add_argument("--pipeline", required=True, choices=["svm", "boosting", "dl"])
    p.add_argument("--run-dir", required=True, help="Path to the training run output directory")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--shared", default="configs/shared.yaml")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found in {run_dir}")

    # Re-resolve from saved config so overrides are preserved.
    cfg = load_yaml(resolved_cfg_path)
    cfg["eval"] = {"split": args.split}

    if args.pipeline == "dl":
        from vn_tsc.pipelines.dl.train import DLPipeline as P
    elif args.pipeline == "svm":
        from vn_tsc.pipelines.svm.train import SVMPipeline as P  # type: ignore[assignment]
    elif args.pipeline == "boosting":
        from vn_tsc.pipelines.boosting.train import BoostingPipeline as P  # type: ignore[assignment]
    else:
        raise ValueError(args.pipeline)

    metrics = P(cfg, run_dir).evaluate()
    print(f"[{args.pipeline}] eval ({args.split}):", metrics)


if __name__ == "__main__":
    main()
