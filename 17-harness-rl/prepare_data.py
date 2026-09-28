#!/usr/bin/env python3
"""Prepare this chapter's shared data, using the versioned central downloader."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT/"scripts/prepare_all.py"), *sys.argv[1:]],check=True,cwd=ROOT)
