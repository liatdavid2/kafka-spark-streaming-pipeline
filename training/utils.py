from datetime import datetime
from pathlib import Path


def make_model_version_path(models_dir: Path, prefix: str = "intrusion_model") -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return models_dir / f"{prefix}_{timestamp}.joblib"