import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).resolve().parent
README_PATH = ROOT_DIR / "README.md"
BASE_VERSION = "0.1.0.dev0"


def get_long_description() -> str:
    if README_PATH.exists():
        return README_PATH.read_text(encoding="utf-8")
    return "Standalone development packaging for FlashMask linear_attn."


def get_package_version() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return BASE_VERSION
    return f"{BASE_VERSION}+g{commit}"


def get_packages() -> list[str]:
    return find_packages(include=["linear_attn", "linear_attn.*"])


def build_setup_kwargs() -> dict:
    return {
        "name": "flashmask-linear-attn-dev",
        "version": get_package_version(),
        "description": "Standalone development package for FlashMask linear attention ops",
        "long_description": get_long_description(),
        "long_description_content_type": "text/markdown",
        "author": "PaddlePaddle Authors",
        "packages": get_packages(),
        "python_requires": ">=3.10",
        "install_requires": [
            "einops",
            "typing_extensions",
        ],
        "extras_require": {
            "test": ["pytest"],
        },
    }


def main():
    setup(**build_setup_kwargs())


if __name__ == "__main__":
    main()
