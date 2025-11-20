"""Layered configuration handling for the AI trading bot project.

Per AGENTS.md, configuration follows a strict hierarchy:
1. Python defaults defined here (safe for version control)
2. Editable YAML overrides in `config/trading_config.yaml`
3. Environment variables using the `AI_TRADING_` prefix (ideal for secrets)
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
import os
import yaml

# Load environment variables from a .env file if present.
load_dotenv()

ENV_PREFIX = "AI_TRADING_"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "trading_config.yaml"
TRAINING_DATA_ROOT = Path(
    os.getenv(f"{ENV_PREFIX}TRAINING_DATA", "/Users/gervaciusjr/Desktop/AI Trading Bot/Training data")
).expanduser()

T = TypeVar("T")


def env(key: str, default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    """Fetch an environment variable, optionally enforcing its presence."""

    value = os.getenv(key, default)
    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def env_bool(key: str, default: bool = False) -> bool:
    """Boolean convenience accessor for environment variables."""

    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_path(path_like: Optional[str | Path], *, default: Path) -> Path:
    if path_like is None:
        return default
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class DataPaths:
    """Rooted directory structure for data artifacts."""

    root: Path = PROJECT_ROOT / "data"
    external_training: Path = TRAINING_DATA_ROOT
    raw: Path = field(init=False)
    processed: Path = field(init=False)
    features: Path = field(init=False)
    checkpoints: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", self.root / "raw")
        object.__setattr__(self, "processed", self.root / "processed")
        object.__setattr__(self, "features", self.root / "features")
        object.__setattr__(self, "checkpoints", self.root / "checkpoints")


@dataclass(frozen=True)
class ModelHyperparameters:
    sequence_length: int = 256
    feature_count: int = 168
    hidden_size: int = 512
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 64


@dataclass(frozen=True)
class TradingParameters:
    instrument: str = "EURUSD"
    base_timeframe: str = "1m"
    risk_per_trade: float = 0.01
    max_daily_drawdown: float = 0.02
    datasource_root: Path = TRAINING_DATA_ROOT


@dataclass(frozen=True)
class FeatureEngineeringSettings:
    max_cache_size: int = 3
    ensure_temporal_order: bool = True
    default_timeframes: Tuple[str, ...] = ("1m", "5m", "15m", "1h")
    wick_window: int = 50
    structure_lookback: int = 200


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs" / "training"
    rotation: str = "50 MB"
    retention: str = "14 days"


@dataclass(frozen=True)
class Settings:
    environment: str
    data_paths: DataPaths
    model: ModelHyperparameters
    trading: TradingParameters
    feature_engineering: FeatureEngineeringSettings
    logging: LoggingSettings
    extra: Dict[str, Any]


def load_trading_config(path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load the YAML trading configuration if available."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_dataclass(
    cls: Type[T], overrides: Optional[Dict[str, Any]], transforms: Optional[Dict[str, Callable[[Any], Any]]] = None
) -> T:
    overrides = overrides or {}
    transforms = transforms or {}
    base = cls()
    kwargs: Dict[str, Any] = {}
    for field_info in fields(cls):
        value = overrides.get(field_info.name, getattr(base, field_info.name))
        if field_info.name in transforms:
            value = transforms[field_info.name](value)
        kwargs[field_info.name] = value
    return cls(**kwargs)


def _build_settings() -> Settings:
    config = load_trading_config()
    environment = os.getenv(f"{ENV_PREFIX}ENV", config.get("environment", "development"))

    data_root = _resolve_path(config.get("data", {}).get("root"), default=PROJECT_ROOT / "data")
    external_data = _resolve_path(config.get("trading", {}).get("datasource_root"), default=TRAINING_DATA_ROOT)
    data_paths = DataPaths(root=data_root, external_training=external_data)

    model = _build_dataclass(ModelHyperparameters, config.get("model"))
    trading_overrides = config.get("trading", {}) | {"datasource_root": external_data}
    trading = _build_dataclass(
        TradingParameters,
        trading_overrides,
        transforms={"datasource_root": lambda value: _resolve_path(value, default=external_data)},
    )
    feature_settings = _build_dataclass(
        FeatureEngineeringSettings,
        config.get("feature_engineering"),
        transforms={
            "default_timeframes": lambda value: tuple(value)
            if value
            else FeatureEngineeringSettings().default_timeframes
        },
    )

    logging_overrides = config.get("logging", {}).copy()
    env_log_level = os.getenv(f"{ENV_PREFIX}LOG_LEVEL")
    if env_log_level:
        logging_overrides["level"] = env_log_level
    logging_settings = _build_dataclass(
        LoggingSettings,
        logging_overrides,
        transforms={"log_dir": lambda value: _resolve_path(value, default=LoggingSettings().log_dir)},
    )

    known_keys = {"environment", "data", "model", "trading", "feature_engineering", "logging"}
    extra = {k: v for k, v in config.items() if k not in known_keys}

    return Settings(
        environment=environment,
        data_paths=data_paths,
        model=model,
        trading=trading,
        feature_engineering=feature_settings,
        logging=logging_settings,
        extra=extra,
    )


_SETTINGS_CACHE: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Return cached settings (or reload if requested)."""

    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or reload:
        _SETTINGS_CACHE = _build_settings()
    return _SETTINGS_CACHE


__all__ = [
    "CONFIG_FILE",
    "DataPaths",
    "FeatureEngineeringSettings",
    "LoggingSettings",
    "ModelHyperparameters",
    "Settings",
    "TradingParameters",
    "env",
    "env_bool",
    "get_settings",
    "load_trading_config",
]
