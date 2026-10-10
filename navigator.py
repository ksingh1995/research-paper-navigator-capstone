"""Portable CLI for the final system, saved demonstrations, and evidence summaries."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from research_navigator.cli import main

if __name__ == "__main__":
    main()
