from __future__ import annotations
import argparse
from vn_tsc.runtime.pack import zip_run_dir

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    print("wrote", zip_run_dir(args.run_dir))

if __name__ == "__main__":
    main()
