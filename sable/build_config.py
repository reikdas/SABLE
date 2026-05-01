from __future__ import annotations

import os
from enum import Enum


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class CSRKernel(str, Enum):
    MKL = "mkl"
    SPREG = "spreg"
    SPV8 = "spv8"
    UZP = "uzp"
    NAIVE = "naive"


class VBRKernel(str, Enum):
    NAIVE = "blocknaive"
    MIXED = "blockmixed"
    MKL = "blockmkl"

    @classmethod
    def _missing_(cls, value):
        if value == "blocksmixed":
            return cls.MIXED
        return None


class VDIAKernel(str, Enum):
    NAIVE = "bandnaive"


def _detect_mkl_config() -> tuple[list[str], bool]:
    """Detect MKL compiler flags and availability."""
    mklroot = os.environ.get("MKLROOT")
    if mklroot and os.path.isdir(os.path.join(mklroot, "include")):
        return [f"-I{mklroot}/include", f"-L{mklroot}/lib/intel64", "-lmkl_rt"], True

    nix_cflags = os.environ.get("NIX_CFLAGS_COMPILE", "")
    nix_ldflags = os.environ.get("NIX_LDFLAGS", "")
    if any("mkl" in token.lower() for token in (nix_cflags.split() + nix_ldflags.split())):
        return ["-lmkl_rt"], True

    system_mkl_include_dirs = [
        "/usr/include",
        "/usr/include/mkl",
        "/usr/include/x86_64-linux-gnu",
    ]
    for inc_dir in system_mkl_include_dirs:
        if os.path.isfile(os.path.join(inc_dir, "mkl.h")):
            flags = ["-lmkl_rt"]
            if inc_dir != "/usr/include":
                flags = [f"-I{inc_dir}"] + flags
            return flags, True

    mkl_path = os.path.join("/home", "min", "a", "das160", "intel", "oneapi", "mkl", "latest")
    flags = [f"-I{mkl_path}/include", f"-L{mkl_path}/lib/intel64", "-lmkl_rt"]
    return flags, os.path.isdir(os.path.join(mkl_path, "include"))


MKL_FLAGS, MKL_AVAILABLE = _detect_mkl_config()

CFLAGS = [
    "-O3",
    "-march=native",
    "-funroll-all-loops",
    "-mprefer-vector-width=512",
    "-mavx",
    "-ffast-math",
    "-fopenmp",
    "-lpthread",
]
