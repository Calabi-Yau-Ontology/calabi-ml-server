from dataclasses import dataclass
from src.config import get_settings

@dataclass
class NLPConfig:
    model_dir: str

def get_nlp_config() -> NLPConfig:
    settings = get_settings()
    return NLPConfig(
        model_dir=settings.MODEL_DIR,
    )
