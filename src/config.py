"""Configuration management for Stock/Crypto Recommender."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the recommender system."""

    _instance = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        """Singleton pattern for configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables."""
        # Load environment variables
        load_dotenv()

        # Find config file
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._get_defaults()

        # Override with environment variables
        self._apply_env_overrides()

    def _find_config_file(self) -> Path | None:
        """Find the configuration file."""
        possible_paths = [
            Path("config/settings.yaml"),
            Path("settings.yaml"),
            Path.home() / ".stock-recommender" / "settings.yaml",
        ]

        # Also check relative to this file
        src_dir = Path(__file__).parent
        project_root = src_dir.parent
        possible_paths.insert(0, project_root / "config" / "settings.yaml")

        for path in possible_paths:
            if path.exists():
                return path
        return None

    def _get_defaults(self) -> dict[str, Any]:
        """Get default configuration values."""
        return {
            "analyst": {
                "min_analysts": 10,
                "max_analysts": 15,
                "horizon_months_min": 6,
                "horizon_months_max": 12,
            },
            "data_sources": {
                "analyst_ratings": True,
                "betting_markets": True,
                "news_aggregation": True,
            },
            "betting_markets": {
                "polymarket": {
                    "enabled": True,
                    "base_url": "https://gamma-api.polymarket.com",
                }
            },
            "news": {
                "rss_feeds": [],
                "max_age_days": 7,
                "max_articles_per_source": 20,
            },
            "recommendation": {
                "analyst_weight": 0.5,
                "betting_weight": 0.25,
                "news_weight": 0.25,
                "strong_buy_threshold": 0.8,
                "buy_threshold": 0.6,
                "hold_threshold": 0.4,
            },
            "rate_limits": {
                "requests_per_minute": 60,
                "retry_attempts": 3,
                "retry_delay_seconds": 2,
            },
            "cache": {"enabled": True, "ttl_minutes": 30},
            "output": {"verbose": True, "top_n": 10},
        }

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "RECOMMENDER_MIN_ANALYSTS": ("analyst", "min_analysts", int),
            "RECOMMENDER_MAX_ANALYSTS": ("analyst", "max_analysts", int),
            "RECOMMENDER_ANALYST_WEIGHT": ("recommendation", "analyst_weight", float),
            "RECOMMENDER_BETTING_WEIGHT": ("recommendation", "betting_weight", float),
            "RECOMMENDER_NEWS_WEIGHT": ("recommendation", "news_weight", float),
            "RECOMMENDER_TOP_N": ("output", "top_n", int),
            "RECOMMENDER_VERBOSE": ("output", "verbose", lambda x: x.lower() == "true"),
        }

        for env_var, (section, key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = converter(value)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            *keys: Configuration keys to traverse
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def analyst_min_count(self) -> int:
        """Minimum number of analysts for recommendations."""
        return self.get("analyst", "min_analysts", default=10)

    @property
    def analyst_max_count(self) -> int:
        """Maximum number of analysts to consider."""
        return self.get("analyst", "max_analysts", default=15)

    @property
    def horizon_months(self) -> tuple[int, int]:
        """Investment horizon range in months."""
        return (
            self.get("analyst", "horizon_months_min", default=6),
            self.get("analyst", "horizon_months_max", default=12),
        )

    @property
    def analyst_weight(self) -> float:
        """Weight for analyst ratings in final score."""
        return self.get("recommendation", "analyst_weight", default=0.5)

    @property
    def betting_weight(self) -> float:
        """Weight for betting market data in final score."""
        return self.get("recommendation", "betting_weight", default=0.25)

    @property
    def news_weight(self) -> float:
        """Weight for news sentiment in final score."""
        return self.get("recommendation", "news_weight", default=0.25)

    @property
    def top_n(self) -> int:
        """Number of top recommendations to show."""
        return self.get("output", "top_n", default=10)

    @property
    def verbose(self) -> bool:
        """Whether to show verbose output."""
        return self.get("output", "verbose", default=True)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Global config instance
config = Config()
