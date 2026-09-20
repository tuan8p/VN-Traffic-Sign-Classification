from __future__ import annotations
import argparse
from pathlib import Path
from vn_tsc.utils.io import load_yaml

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True, choices=["svm", "boosting", "dl"])
    p.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    _ = load_yaml(run_dir / "resolved_config.yaml")
    raise NotImplementedError(f"TODO(team-{args.pipeline}): eval from {run_dir}")

if __name__ == "__main__":
    main()
