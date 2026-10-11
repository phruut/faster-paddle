"""Convert ONNX models to OpenVINO IR with the pinned openvino package.

Usage:
    python scripts/convert_ir.py OUT_DIR [--flat] SIZE [SIZE ...]

Default layout (what the OV backend's OV_IR_DIR expects):
    OUT_DIR/{size}/{det,rec}.xml + .bin
--flat writes release-asset names instead:
    OUT_DIR/ppocrv6_{size}_{det,rec}.xml + .bin
"""
import argparse
from pathlib import Path

import openvino as ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("sizes", nargs="+")
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for stage in ("det", "rec"):
            model = ov.convert_model(f"models/{size}/{stage}.onnx")
            dest = args.out_dir if args.flat else args.out_dir / size
            dest.mkdir(parents=True, exist_ok=True)
            name = f"ppocrv6_{size}_{stage}.xml" if args.flat else f"{stage}.xml"
            ov.save_model(model, str(dest / name))
            print("converted", size, stage, flush=True)


if __name__ == "__main__":
    main()
