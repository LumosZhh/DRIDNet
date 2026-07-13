"""DRIDNet command-line entry point."""

from __future__ import annotations
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runner import DRIDNetRunner

if __name__ == "__main__":
    DRIDNetRunner().run()
