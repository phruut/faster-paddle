#!/usr/bin/env python3
"""Verify a built faster-paddle OpenVINO wheel vendors its native runtime.

The OV wheels must be self-contained: build.rs stages the pinned OpenVINO
runtime into the package directory, where the OS loader finds it beside the
extension with no PATH/LD_LIBRARY_PATH setup. This script is the CI proof that
those files actually made it into the wheel (run after the wheel build).

The platform comes from the wheel tag; required files are matched by their
vendoring prefix because OpenVINO release numbers live in the filenames.

Usage: python scripts/check_wheel_runtime.py dist/*.whl
Exits 0 on success, 1 (with a GitHub ::error:: line) on any missing file.
"""

import glob
import os
import re
import sys
import zipfile

PLATFORM_PATTERNS = {
    "windows": [
        r"^openvino\.dll$",
        r"^openvino_c\.dll$",
        r"^openvino_ir_frontend\.dll$",
        r"^tbb12\.dll$",
        r"^tbbbind_2_5\.dll$",
        r"^openvino_(intel|arm)_cpu_plugin\.dll$",
        r"^plugins\.xml$",
    ],
    "linux": [
        r"^libopenvino\.so\.",
        r"^libopenvino_c\.so\.",
        r"^libopenvino_ir_frontend\.so\.",
        r"^libtbb\.so\.12",
        r"^libtbbbind_2_5\.so\.",
        r"^libhwloc\.so\.",
        r"^libopenvino_(intel|arm)_cpu_plugin\.so$",
        r"^plugins\.xml$",
    ],
    "macos": [
        r"^libopenvino\.",
        r"^libopenvino_c\.",
        r"^libopenvino_ir_frontend\.",
        r"^libtbb\.12",
        r"^libtbbbind_2_5\.",
        r"^libhwloc\.",
        r"^libopenvino_(intel|arm)_cpu_plugin\.so$",
        r"^plugins\.xml$",
    ],
}

EXTENSION = re.compile(r"^faster_paddle/faster_paddle.*\.(so|pyd)$")


def platform_of(wheel_path):
    tag = os.path.basename(wheel_path).rsplit("-", 1)[-1]
    if tag.endswith(".whl"):
        tag = tag[: -len(".whl")]
    if "win" in tag:
        return "windows"
    if "macosx" in tag:
        return "macos"
    if "linux" in tag:
        return "linux"
    raise SystemExit(
        f"::error::cannot deduce platform from wheel {os.path.basename(wheel_path)!r}"
    )


def check(wheel_path):
    """Return (missing patterns, package member names) for one wheel."""
    platform = platform_of(wheel_path)
    names = zipfile.ZipFile(wheel_path).namelist()
    package = [n for n in names if n.startswith("faster_paddle/")]
    missing = []
    if not any(EXTENSION.match(n) for n in names):
        missing.append("extension module (*.pyd/*.so)")
    for pattern in PLATFORM_PATTERNS[platform]:
        if not any(re.search(pattern, n.split("/", 1)[-1]) for n in package):
            missing.append(pattern)
    return missing, package


def main(argv):
    wheels = [p for arg in argv for p in glob.glob(arg)]
    if not wheels:
        print("::error::no wheel matched the given arguments")
        return 1
    failed = False
    for wheel in wheels:
        tag = os.path.basename(wheel)
        missing, package = check(wheel)
        if missing:
            failed = True
            print(
                "::error::vendored OpenVINO runtime incomplete in "
                f"{tag}: missing {', '.join(missing)}"
            )
            print(f"--- {tag} package contents ---")
            for name in package:
                print(f"  {name}")
        else:
            print(
                f"vendored runtime OK: {tag} "
                f"({platform_of(wheel)}, {len(package)} package files)"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
