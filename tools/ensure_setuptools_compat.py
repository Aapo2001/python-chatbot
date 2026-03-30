"""Ensure setuptools is present in a range that works for local builds."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata

TARGET_SPEC = "setuptools>=69.5,<80"


def _parse_version(version: str) -> tuple[int, ...]:
    match = re.match(r"^\d+(?:\.\d+)*", version)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def _is_compatible(version: str) -> bool:
    parsed = _parse_version(version)
    return parsed >= (69, 5) and parsed < (80,)


def _install_target() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", TARGET_SPEC]
    )


def main() -> None:
    try:
        version = metadata.version("setuptools")
    except metadata.PackageNotFoundError:
        print(f"[pixi] setuptools is missing, installing {TARGET_SPEC}...")
        _install_target()
        return

    if _is_compatible(version):
        print(f"[pixi] setuptools {version} is already compatible.")
        return

    print(
        f"[pixi] setuptools {version} is outside the supported range, installing {TARGET_SPEC}..."
    )
    _install_target()


if __name__ == "__main__":
    main()
