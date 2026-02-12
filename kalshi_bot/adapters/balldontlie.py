from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class NbaGame:
    id: int
    date: str
    home: Optional[str]
    away: Optional[str]
    status: Optional[str]


class BallDontLieClient:
    """
    balldontlie.io API
    Base URL: https://api.balldontlie.io/<sport>/v1 (e.g., /v1, /mma/v1, /mls/v1)
    Auth header: Authorization: <API_KEY>
    """

    def __init__(self, api_key: str, base: str = "https://api.balldontlie.io/v1", timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.base = base
        self.timeout = timeout
        self._team_cache: dict[str, List[Dict[str, Any]]] = {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params or {}, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def search_teams(self, name: str) -> List[Dict[str, Any]]:
        key = name.strip().lower()
        if key in self._team_cache:
            return self._team_cache[key]
        data = self._get("/teams", params={"search": name})
        teams = data.get("data", []) or []
        self._team_cache[key] = teams
        return teams

    def games_on_date(self, day: date, team_id: Optional[int] = None) -> List[NbaGame]:
        params: Dict[str, Any] = {"dates[]": day.isoformat()}
        if team_id is not None:
            params["team_ids[]"] = team_id
        data = self._get("/games", params=params)
        games = []
        for g in data.get("data", []) or []:
            games.append(
                NbaGame(
                    id=int(g.get("id")),
                    date=g.get("date") or "",
                    home=(g.get("home_team") or {}).get("full_name"),
                    away=(g.get("visitor_team") or {}).get("full_name"),
                    status=g.get("status"),
                )
            )
        return games
