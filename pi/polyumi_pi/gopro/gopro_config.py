"""Persistent GoPro connection config saved in ~/.polyumi/gopro_config.json."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

GOPRO_CONFIG_PATH = Path.home() / '.polyumi' / 'gopro_config.json'


@dataclass
class GoProConfig:
    """Saved BLE connection info for a GoPro device."""

    name: str
    mac_address: str
    identifier: str  # last 4 digits of GoPro serial, used as BLE scan token


def load_gopro_config() -> GoProConfig | None:
    """Load config from disk, returning None if not found or malformed."""
    if not GOPRO_CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(GOPRO_CONFIG_PATH.read_text())
        return GoProConfig(**data)
    except Exception:
        return None


def save_gopro_config(config: GoProConfig) -> None:
    """Write config to disk, creating parent directories as needed."""
    GOPRO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOPRO_CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2))
