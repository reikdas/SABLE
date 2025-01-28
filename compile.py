#!/usr/bin/env python3
import os
import subprocess

USE_SPLIT=True
SPLITTER_DIR="sable-splitter"

def compile(basedir, fname, coreid, codegen_dir, timeout):
    if USE_SPLIT:
        parts=64
    else:
        parts = 0
    if not os.path.isdir(SPLITTER_DIR):
        subprocess.run(["git", "clone", "https://github.com/ulysses4ever/sable-splitter"])

    subprocess.run([
        "taskset",
        "-a",
        "-c",
        str(coreid),
        SPLITTER_DIR + "/split.sh",
        str(parts),
        codegen_dir + "/" + fname + ".c"],
                   cwd=basedir, check=True, capture_output=True, text=True, timeout=timeout)

    return basedir + "/" + fname + "/" + fname
