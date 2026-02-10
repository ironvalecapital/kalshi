from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitConfig(BaseModel):
    tier: Literal["basic", "advanced", "premier", "prime"] = "basic"
    read_per_sec: int = 20
    write_per_sec: int = 10
    burst: int = 40


class RiskLimits(BaseModel):
    max_daily_loss_usd: float = 100.0
    max_position_per_market_usd: float = 200.0
    max_open_orders: int = 10
    max_order_size_contracts: int = 20
    max_notional_usd: float = 500.0
    max_trade_freq_per_market_seconds: int = 10
    cooldown_seconds_after_loss: int = 300
    max_consecutive_losses: int = 3
    pause_on_spread_wide: bool = True
    spread_wide_threshold_cents: float = 10.0


class ExecutionConfig(BaseModel):
    prefer_maker: bool = True
    min_order_interval_seconds: int = 5
    require_edge_cents: float = 2.0
    maker_fee_rate: float = 0.0175
    taker_fee_rate: float = 0.07
    max_cancels_per_minute: int = 10


class DataConfig(BaseModel):
    env: Literal["demo", "prod"] = "demo"
    api_base_demo: str = "https://demo-api.kalshi.co/trade-api/v2"
    api_base_prod: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url_demo: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    ws_url_prod: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"


class WeatherConfig(BaseModel):
    forecast_sigma_f: float = 2.0
    min_fill_prob: float = 0.25
    max_spread_cents: float = 12.0
    min_edge_after_fees_cents: float = 1.5
    min_time_to_close_hours: float = 2.0
    max_time_to_close_hours: float = 48.0
    w_trades: float = 1.0
    w_volume: float = 0.1
    w_spread: float = 0.05
    min_trades_24h: int = 0
    top_n: int = 200
    bid_improve_cents: int = 1
    max_depth_levels: int = 3
    allow_cross_spread: bool = False
    min_rr: float = 1.5
    max_order_size: int = 10
    ev_size_scale_cents: float = 3.0
    depth_fill_weight: float = 0.005
    trades_fill_weight: float = 0.02
    regime_t1_hours: float = 24.0
    regime_t2_hours: float = 2.0
    near_close_min_rr: float = 2.0
    near_close_min_ev_cents: float = 2.5
    keywords: list = [
        "WEATHER",
        "TEMP",
        "TEMPERATURE",
        "HIGH TEMPERATURE",
        "PRECIP",
        "SNOW",
        "RAIN",
    ]
    market_overrides: dict = {}
    default_city: str = "NYC"
    city_map: dict = {
        "NYC": {"lat": 40.7128, "lon": -74.0060},
        "CHI": {"lat": 41.8781, "lon": -87.6298},
        "DC": {"lat": 38.9072, "lon": -77.0369},
        "LA": {"lat": 34.0522, "lon": -118.2437},
        "BOS": {"lat": 42.3601, "lon": -71.0589},
        "PHL": {"lat": 39.9526, "lon": -75.1652},
        "ATL": {"lat": 33.7490, "lon": -84.3880},
        "MIA": {"lat": 25.7617, "lon": -80.1918},
        "DEN": {"lat": 39.7392, "lon": -104.9903},
        "SEA": {"lat": 47.6062, "lon": -122.3321},
        "SFO": {"lat": 37.7749, "lon": -122.4194},
        "DAL": {"lat": 32.7767, "lon": -96.7970},
    }


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key_id: Optional[str] = Field(default=None, validation_alias="KALSHI_API_KEY_ID")
    private_key_path: Optional[str] = Field(default=None, validation_alias="KALSHI_PRIVATE_KEY_PATH")
    use_sdk: bool = Field(default=True, validation_alias="KALSHI_USE_SDK")
    db_path: str = Field(default="kalshi.db", validation_alias="KALSHI_DB_PATH")
    log_path: str = Field(default="logs/audit.log", validation_alias="KALSHI_AUDIT_LOG")
    decision_report_dir: str = Field(default="runs", validation_alias="KALSHI_DECISION_REPORT_DIR")
    external_signal_enabled: bool = Field(default=False, validation_alias="KALSHI_EXTERNAL_SIGNALS")
    weather_locations_json: Optional[str] = Field(default=None, validation_alias="WEATHER_LOCATIONS_JSON")
    weather_locations: Optional[str] = Field(default=None, validation_alias="WEATHER_LOCATIONS")
    weather_geojson_path: Optional[str] = Field(default=None, validation_alias="WEATHER_GEOJSON")
    noaa_token: Optional[str] = Field(default=None, validation_alias="NOAA_TOKEN")
    weather_user_agent: str = Field(default="kalshi_decision_engine (contact: ops@example.com)", validation_alias="NWS_USER_AGENT")

    data: DataConfig = DataConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    risk: RiskLimits = RiskLimits()
    execution: ExecutionConfig = ExecutionConfig()
    weather: WeatherConfig = WeatherConfig()


def load_config(path: Optional[str] = None) -> BotSettings:
    settings = BotSettings()
    if path:
        cfg_path = Path(path)
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}
            return BotSettings(**data)
    return settings


def merged_weather_locations(settings: BotSettings) -> dict:
    locations = dict(settings.weather.city_map)
    if settings.weather_locations_json:
        try:
            data = yaml.safe_load(settings.weather_locations_json)
            if isinstance(data, dict):
                locations.update(data)
        except Exception:
            pass
    if settings.weather_locations:
        try:
            data = yaml.safe_load(settings.weather_locations)
            if isinstance(data, dict):
                locations.update(data)
        except Exception:
            pass
    if settings.weather_geojson_path:
        try:
            geo = yaml.safe_load(Path(settings.weather_geojson_path).read_text())
            features = geo.get("features", []) if isinstance(geo, dict) else []
            for feat in features:
                props = feat.get("properties", {})
                code = props.get("code") or props.get("city") or props.get("name")
                coords = feat.get("geometry", {}).get("coordinates", [])
                if code and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    locations[str(code).upper()] = {"lat": float(lat), "lon": float(lon)}
        except Exception:
            pass
    return locations


def api_base(settings: BotSettings) -> str:
    return settings.data.api_base_demo if settings.data.env == "demo" else settings.data.api_base_prod


def ws_url(settings: BotSettings) -> str:
    return settings.data.ws_url_demo if settings.data.env == "demo" else settings.data.ws_url_prod
