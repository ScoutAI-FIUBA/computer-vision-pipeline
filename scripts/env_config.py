from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


load_dotenv()

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
PLAYER_MODEL_ID = os.getenv("PLAYER_MODEL_ID", "basketball-player-detection-3-ycjdo/16")
COURT_MODEL_ID = os.getenv("COURT_MODEL_ID", "basketball-court-detection-2/22")
JERSEY_OCR_MODEL_ID = os.getenv("JERSEY_OCR_MODEL_ID", "")


def get_roboflow_api_key() -> str:
    return os.getenv("ROBOFLOW_API_KEY", "")
