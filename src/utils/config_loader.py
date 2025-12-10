"""Configuration loader utility."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration settings."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader.

        Args:
            config_path: Path to config file. Defaults to config/settings.yaml
        """
        load_dotenv()

        if config_path is None:
            # Find project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config" / "settings.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}

        # Substitute environment variables
        self._substitute_env_vars(self._config)

    def _substitute_env_vars(self, config: Dict[str, Any]) -> None:
        """Recursively substitute environment variables in config."""
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, '')
            elif isinstance(value, dict):
                self._substitute_env_vars(value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'api_keys.fred')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def indicators(self) -> Dict[str, Any]:
        """Get macro indicator configurations."""
        return self.get('indicators', {})

    @property
    def markets(self) -> Dict[str, Any]:
        """Get market instrument configurations."""
        return self.get('markets', {})

    @property
    def analysis_params(self) -> Dict[str, Any]:
        """Get analysis parameters."""
        return self.get('analysis', {})

    @property
    def fred_api_key(self) -> str:
        """Get FRED API key."""
        return self.get('api_keys.fred', os.getenv('FRED_API_KEY', ''))

    @property
    def alpha_vantage_key(self) -> str:
        """Get Alpha Vantage API key."""
        return self.get('api_keys.alpha_vantage', os.getenv('ALPHA_VANTAGE_API_KEY', ''))
