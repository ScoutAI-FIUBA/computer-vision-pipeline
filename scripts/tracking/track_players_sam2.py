#!/usr/bin/env python3
"""Stable SAM2 tracking entrypoint for local video clips."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tracking.track_players_sam2_experiment import main


if __name__ == "__main__":
    main()
