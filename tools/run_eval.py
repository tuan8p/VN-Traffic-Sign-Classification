from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsc.utils.io import load_yaml


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint (DL, SVM, or Boosting)")
    p.add_argument("--pipeline", required=True, choices=["svm", "boosting", "dl"])
    p.add_argument("--run-dir", default=None, help="Path to run output directory (default: latest in outputs/<pipeline>)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--shared", default="configs/shared.yaml")
    p.add_argument("--data-root", default=None, help="Override data.processed_root for evaluation")
    p.add_argument("--processed-root", default=None, help="Alias for --data-root")
    args = p.parse_args(argv)

    if args.run_dir is None or args.run_dir == "latest":
        runs = sorted(Path(f"outputs/{args.pipeline}").glob("*_*"))
        if not runs:
            raise FileNotFoundError(f"No run directories found under outputs/{args.pipeline}")
        run_dir = runs[-1]
        print(f"Auto-selected latest run_dir: {run_dir}")
    else:
        run_dir = Path(args.run_dir)

    resolved_cfg_path = run_dir / "resolved_config.yaml"
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found in {run_dir}")

    cfg = load_yaml(resolved_cfg_path)
    cfg.setdefault("eval", {})["split"] = args.split

    data_root = args.data_root or args.processed_root
    if data_root:
        cfg.setdefault("data", {})["processed_root"] = data_root
        cfg.setdefault("data_source", {})["processed_root"] = data_root

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
