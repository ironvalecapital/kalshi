from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class SportsEvent:
    id: str
    name: str
    date: str
    time: Optional[str]
    home: Optional[str]
    away: Optional[str]
    league: Optional[str]


class SportsDBClient:
    """
    TheSportsDB free API client (v1).
    Free key is "123" per TheSportsDB docs.
    Base URL: https://www.thesportsdb.com/api/v1/json
    Authentication: API key in URL path for v1.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        self.api_key = api_key or "123"
        self.base = "https://www.thesportsdb.com/api/v1/json"
        self.timeout = timeout
        self._team_cache: dict[str, List[Dict[str, Any]]] = {}
        self._next_cache: dict[str, List[Dict[str, Any]]] = {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base}/{self.api_key}/{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json()

    def search_teams(self, team: str) -> List[Dict[str, Any]]:
        key = team.strip().lower()
        if key in self._team_cache:
            return self._team_cache[key]
        data = self._get("searchteams.php", params={"t": team})
        teams = data.get("teams", []) or []
        self._team_cache[key] = teams
        return teams

    def events_next(self, team_id: str) -> List[Dict[str, Any]]:
        if team_id in self._next_cache:
            return self._next_cache[team_id]
        data = self._get("eventsnext.php", params={"id": team_id})
        events = data.get("events", []) or []
        self._next_cache[team_id] = events
        return events

    def events_day(self, day: date, sport: str) -> List[SportsEvent]:
        data = self._get("eventsday.php", params={"d": day.isoformat(), "s": sport})
        events = []
        for e in data.get("events", []) or []:
            events.append(
                SportsEvent(
                    id=str(e.get("idEvent") or ""),
                    name=e.get("strEvent") or "",
                    date=e.get("dateEvent") or "",
                    time=e.get("strTime"),
                    home=e.get("strHomeTeam"),
                    away=e.get("strAwayTeam"),
                    league=e.get("strLeague"),
                )
            )
        return events
