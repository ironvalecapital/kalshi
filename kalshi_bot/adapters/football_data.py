from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class FootballMatch:
    id: int
    utc_date: str
    home: Optional[str]
    away: Optional[str]
    competition: Optional[str]
    status: Optional[str]


class FootballDataClient:
    """
    football-data.org API v4
    Base URL: https://api.football-data.org/v4
    Auth header: X-Auth-Token
    """

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.base = "https://api.football-data.org/v4"
        self.timeout = timeout
        self._team_cache: dict[str, List[Dict[str, Any]]] = {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        headers = {"X-Auth-Token": self.api_key}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params or {}, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def search_teams(self, name: str) -> List[Dict[str, Any]]:
        key = name.strip().lower()
        if key in self._team_cache:
            return self._team_cache[key]
        data = self._get("/teams", params={"name": name})
        teams = data.get("teams", []) or []
        self._team_cache[key] = teams
        return teams

    def matches_on_date(self, day: date, team_id: Optional[int] = None) -> List[FootballMatch]:
        params = {"dateFrom": day.isoformat(), "dateTo": day.isoformat()}
        if team_id is not None:
            params["team"] = team_id
        data = self._get("/matches", params=params)
        matches = []
        for m in data.get("matches", []) or []:
            matches.append(
                FootballMatch(
                    id=int(m.get("id")),
                    utc_date=m.get("utcDate") or "",
                    home=(m.get("homeTeam") or {}).get("name"),
                    away=(m.get("awayTeam") or {}).get("name"),
                    competition=(m.get("competition") or {}).get("name"),
                    status=m.get("status"),
                )
            )
        return matches
