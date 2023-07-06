from pathlib import Path

# Paths
ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = Path("/data/ubuntu/data/dataset")  # ROOT_PATH / "data"
CONFIG_PATH = ROOT_PATH / "src" / "config" / "training_config.yaml"
