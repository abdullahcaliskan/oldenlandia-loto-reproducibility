#!/usr/bin/env python3
"""Entry point for the accepted-manuscript analysis pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reporting import main

if __name__ == "__main__":
    main()
