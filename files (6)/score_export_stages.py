"""Score original, MIRNet, and each cumulative post-processing stage."""
import argparse
from pathlib import Path

from quality_metrics import score_export_stages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()
    score_export_stages(args.export_root)
