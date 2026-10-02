"""Project configuration."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED = 42
FRAUD_RATE = 0.035
