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
    bankroll_usd: float = 5.0
    kelly_fraction: float = 0.33
    kelly_use_fill_prob: bool = True


class DataConfig(BaseModel):
    env: Literal["demo", "prod"] = "demo"
    api_base_demo: str = "https://demo-api.kalshi.co/trade-api/v2"
    api_base_prod: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url_demo: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    ws_url_prod: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"


class WeatherConfig(BaseModel):
    forecast_sigma_f: float = 2.0
    min_fill_prob: float = 0.005
    max_spread_cents: float = 80.0
    min_edge_after_fees_cents: float = 0.01
    entry_edge_pp: float = 5.0
    exit_edge_pp: float = 45.0
    min_time_to_close_hours: float = 0.5
    max_time_to_close_hours: float = 168.0
    use_close_window: bool = False
    statuses: list = ["open", "initialized"]
    w_trades: float = 1.0
    w_volume: float = 0.1
    w_spread: float = 0.05
    min_trades_24h: int = 0
    top_n: int = 300
    allow_unmatched_markets: bool = True
    bid_improve_cents: int = 1
    max_depth_levels: int = 3
    allow_cross_spread: bool = True
    min_rr: float = 1.01
    max_order_size: int = 25
    ev_size_scale_cents: float = 3.0
    depth_fill_weight: float = 0.005
    trades_fill_weight: float = 0.02
    regime_t1_hours: float = 24.0
    regime_t2_hours: float = 1.0
    near_close_min_rr: float = 1.01
    near_close_min_ev_cents: float = 0.1
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
    allowed_cities: list = []
    max_trades_per_cycle: int = 5
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
        "ALBANY": {"lat": 42.6526, "lon": -73.7562},
        "ANNAPOLIS": {"lat": 38.9784, "lon": -76.4922},
        "ATLANTA": {"lat": 33.7490, "lon": -84.3880},
        "AUSTIN": {"lat": 30.2672, "lon": -97.7431},
        "BATON ROUGE": {"lat": 30.4515, "lon": -91.1871},
        "BISMARCK": {"lat": 46.8083, "lon": -100.7837},
        "BOISE": {"lat": 43.6150, "lon": -116.2023},
        "BOSTON": {"lat": 42.3601, "lon": -71.0589},
        "CARSON CITY": {"lat": 39.1638, "lon": -119.7674},
        "CHARLESTON": {"lat": 38.3498, "lon": -81.6326},
        "CHEYENNE": {"lat": 41.1400, "lon": -104.8202},
        "COLUMBIA": {"lat": 34.0007, "lon": -81.0348},
        "COLUMBUS": {"lat": 39.9612, "lon": -82.9988},
        "CONCORD": {"lat": 43.2081, "lon": -71.5376},
        "DENVER": {"lat": 39.7392, "lon": -104.9903},
        "DES MOINES": {"lat": 41.5868, "lon": -93.6250},
        "DOVER": {"lat": 39.1582, "lon": -75.5244},
        "FRANKFORT": {"lat": 38.2009, "lon": -84.8733},
        "HARRISBURG": {"lat": 40.2732, "lon": -76.8867},
        "HARTFORD": {"lat": 41.7658, "lon": -72.6734},
        "HELENA": {"lat": 46.5891, "lon": -112.0391},
        "HONOLULU": {"lat": 21.3069, "lon": -157.8583},
        "INDIANAPOLIS": {"lat": 39.7684, "lon": -86.1581},
        "JACKSON": {"lat": 32.2988, "lon": -90.1848},
        "JEFFERSON CITY": {"lat": 38.5767, "lon": -92.1735},
        "JUNEAU": {"lat": 58.3019, "lon": -134.4197},
        "LANSING": {"lat": 42.7325, "lon": -84.5555},
        "LINCOLN": {"lat": 40.8136, "lon": -96.7026},
        "LITTLE ROCK": {"lat": 34.7465, "lon": -92.2896},
        "MADISON": {"lat": 43.0731, "lon": -89.4012},
        "MONTGOMERY": {"lat": 32.3792, "lon": -86.3077},
        "MONTPELIER": {"lat": 44.2601, "lon": -72.5754},
        "NASHVILLE": {"lat": 36.1627, "lon": -86.7816},
        "OKLAHOMA CITY": {"lat": 35.4676, "lon": -97.5164},
        "OLYMPIA": {"lat": 47.0379, "lon": -122.9007},
        "PHOENIX": {"lat": 33.4484, "lon": -112.0740},
        "PIERRE": {"lat": 44.3683, "lon": -100.3509},
        "PROVIDENCE": {"lat": 41.8240, "lon": -71.4128},
        "RALEIGH": {"lat": 35.7796, "lon": -78.6382},
        "RICHMOND": {"lat": 37.5407, "lon": -77.4360},
        "SACRAMENTO": {"lat": 38.5816, "lon": -121.4944},
        "SAINT PAUL": {"lat": 44.9537, "lon": -93.0900},
        "ST PAUL": {"lat": 44.9537, "lon": -93.0900},
        "SALEM": {"lat": 44.9429, "lon": -123.0351},
        "SANTA FE": {"lat": 35.6870, "lon": -105.9378},
        "SPRINGFIELD": {"lat": 39.7817, "lon": -89.6501},
        "TALLAHASSEE": {"lat": 30.4383, "lon": -84.2807},
        "TOPEKA": {"lat": 39.0473, "lon": -95.6752},
        "TRENTON": {"lat": 40.2204, "lon": -74.7643},
        "HARTFORD CT": {"lat": 41.7658, "lon": -72.6734},
        "AUGUSTA": {"lat": 44.3106, "lon": -69.7795},
        "AUGUSTA ME": {"lat": 44.3106, "lon": -69.7795},
        "AUGUSTA GA": {"lat": 33.4735, "lon": -82.0105},
        "SAINT LOUIS": {"lat": 38.6270, "lon": -90.1994},
        "NEW ORLEANS": {"lat": 29.9511, "lon": -90.0715},
        "NEW YORK": {"lat": 40.7128, "lon": -74.0060},
    }


class SportsConfig(BaseModel):
    keywords: list = [
        "SPORT",
        "SPORTS",
        "NBA",
        "NFL",
        "NHL",
        "MLB",
        "NCAAB",
        "NCAAF",
        "GAME",
        "MATCH",
        "TEAM",
    ]
    allowlist: list = []
    max_spread_cents: float = 70.0
    min_trades_60m: int = 0
    min_trades_5m: int = 0
    min_top_depth: int = 0
    min_ev_cents: float = 0.05
    extreme_edge_cents: float = 6.0
    urgency_window_sec: int = 300
    min_order_interval_sec: int = 3
    max_cancels_per_min: int = 10
    top_n: int = 50
    base_size: int = 1
    max_order_size: int = 10
    allow_unmatched_markets: bool = True
    statuses: list = ["open", "initialized"]
    yes_longshot_max_cents: int = 10
    no_tail_min_cents: int = 90
    maker_only: bool = True
    category_ev_multiplier: float = 1.25
    max_scan_markets: int = 400
    markets_cache_ttl_sec: int = 90
    auto_pick_use_summary: bool = True
    auto_pick_top_n: int = 100
    daily_report_interval_sec: int = 3600
    stale_order_max_age_sec: int = 300
    stale_order_cancel_edge_cents: float = 0.1
    min_spread_cents: int = 1
    avoid_price_low_cents: int = 2
    avoid_price_high_cents: int = 98
    depth_size_divisor: int = 5
    min_arb_depth: int = 10
    allow_arb_taker: bool = True
    spread_scan_weight: float = 1.0
    simple_active_maker: bool = True
    simple_min_spread_cents: int = 1


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
    sports: SportsConfig = SportsConfig()


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
