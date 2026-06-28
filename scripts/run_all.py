"""Run all released analysis scripts in manuscript order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "00_descriptive_context.py",
    "01_context_baselines.py",
    "02_low_degree_visibility.py",
    "03_path_repertoire_checks.py",
    "04_generate_path_subgraph.py",
    "05_effect_size_and_structure.py",
]


def main():
    script_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print("\n" + "=" * 78)
        print("Running", script)
        print("=" * 78)
        subprocess.check_call([sys.executable, str(script_dir / script)])


if __name__ == "__main__":
    main()
