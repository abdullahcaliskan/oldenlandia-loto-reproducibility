#!/usr/bin/env python3
"""Entry point for the published IEEE Access analysis pipeline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_INPUTS = (
    "s_MTBLS13314.txt",
    "m_MTBLS13314_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv",
    "m_MTBLS13314_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv",
)


def main() -> int:
    missing = [name for name in REQUIRED_INPUTS if not (ROOT / "data_input" / name).is_file()]
    if missing:
        print("Missing required MetaboLights input files in data_input/:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print("Download them from MTBLS13314; see data_input/README.md.", file=sys.stderr)
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from reporting import main as run_analysis
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "pandas", "scipy", "sklearn", "matplotlib"}:
            print("A required Python package is missing. Run: python -m pip install -r requirements.txt", file=sys.stderr)
            return 3
        raise

    run_analysis()
    return 0


if __name__ == "__main__":
    sys.exit(main())
