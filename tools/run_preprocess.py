from __future__ import annotations
import argparse
from vn_tsc.config.resolve import resolve_config
from vn_tsc.data.preprocess_offline import run_offline_preprocess
from vn_tsc.features.classical import build_feature_store

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Offline shared preprocess (no W&B)")
    p.add_argument("--shared", default="configs/shared.yaml")
    p.add_argument("--runtime", default="configs/runtime/local.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--skip-crops", action="store_true",
                   help="reuse an existing crop cache and only rebuild features")
    p.add_argument("--skip-features", action="store_true",
                   help="write the crop cache only (DL needs nothing else)")
    p.add_argument("--no-augment", action="store_true",
                   help="skip offline augmentation (for the with/without ablation)")
    args = p.parse_args(argv)
    overrides: dict = {}
    if args.data_root:
        overrides["data"] = {"root": args.data_root}
    if args.no_augment:
        overrides["augment"] = {"enabled": False}
    cfg = resolve_config(shared_yaml=args.shared, runtime_yaml=args.runtime, overrides=overrides or None)
    root = cfg.get("data", {}).get("root") or ""
    if not root and not args.skip_crops:
        raise SystemExit("Set DATA_ROOT or --data-root")
    if not args.skip_crops:
        run_offline_preprocess(cfg, root, args.out)
    if not args.skip_features:
        build_feature_store(cfg, args.out)

if __name__ == "__main__":
    main()
