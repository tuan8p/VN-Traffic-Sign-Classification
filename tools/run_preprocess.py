from __future__ import annotations
import argparse
from vn_tsc.config.resolve import resolve_config
from vn_tsc.data.preprocess_offline import run_offline_preprocess

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Offline shared preprocess (no W&B)")
    p.add_argument("--shared", default="configs/shared.yaml")
    p.add_argument("--runtime", default="configs/runtime/local.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="data/processed")
    args = p.parse_args(argv)
    overrides = {}
    if args.data_root:
        overrides["data"] = {"root": args.data_root}
    cfg = resolve_config(shared_yaml=args.shared, runtime_yaml=args.runtime, overrides=overrides or None)
    root = cfg.get("data", {}).get("root") or ""
    if not root:
        raise SystemExit("Set DATA_ROOT or --data-root")
    run_offline_preprocess(cfg, root, args.out)

if __name__ == "__main__":
    main()
