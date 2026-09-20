from __future__ import annotations
import argparse
from vn_tsc.eval.plots import curves_from_history_json

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    print("wrote", curves_from_history_json(args.history, args.out_dir))

if __name__ == "__main__":
    main()
