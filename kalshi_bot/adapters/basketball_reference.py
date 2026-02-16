from __future__ import annotations

import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup, Comment


@dataclass
class TeamProfile:
    team: str
    team_abbr: str
    season: int
    off_rtg: Optional[float]
    def_rtg: Optional[float]
    net_rtg: Optional[float]
    pace: Optional[float]
    home_win_pct: Optional[float]
    away_win_pct: Optional[float]
    points_home: Optional[float]
    points_away: Optional[float]
    q4_points_for: Optional[float]
    q4_points_against: Optional[float]
    q4_net: Optional[float]
    red_zone_conversion_proxy: Optional[float]


class BasketballReferenceClient:
    """
    Data collector for Basketball-Reference team-level metrics.
    Scrapes public pages with conservative request pacing.
    """

    def __init__(
        self,
        base_url: str = "https://www.basketball-reference.com",
        timeout: float = 20.0,
        sleep_between_requests: float = 0.35,
        max_retries: int = 6,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep_between_requests = sleep_between_requests
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": "ironvale-kalshi-research-bot/1.0"},
        )

    def close(self) -> None:
        self._client.close()

    def _get_soup(self, path: str) -> BeautifulSoup:
        url = f"{self.base_url}{path}"
        attempt = 0
        backoff = 1.0
        while True:
            resp = self._client.get(url)
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
                time.sleep(self.sleep_between_requests)
                return BeautifulSoup(resp.text, "html.parser")
            if attempt >= self.max_retries:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            wait = None
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = None
            if wait is None:
                wait = backoff + random.uniform(0.0, 0.5)
                backoff = min(15.0, backoff * 2.0)
            time.sleep(max(wait, self.sleep_between_requests))
            attempt += 1

    def _table_from_soup(self, soup: BeautifulSoup, table_id: str):
        table = soup.find("table", {"id": table_id})
        if table is not None:
            return table
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            txt = str(c)
            if f'id="{table_id}"' in txt:
                csoup = BeautifulSoup(txt, "html.parser")
                table = csoup.find("table", {"id": table_id})
                if table is not None:
                    return table
        return None

    @staticmethod
    def _to_float(v: Optional[str]) -> Optional[float]:
        if v is None:
            return None
        t = str(v).strip().replace("%", "")
        if t in {"", "None"}:
            return None
        try:
            return float(t)
        except ValueError:
            return None

    @staticmethod
    def _to_int(v: Optional[str]) -> Optional[int]:
        if v is None:
            return None
        t = str(v).strip()
        if t in {"", "None"}:
            return None
        try:
            return int(float(t))
        except ValueError:
            return None

    def _team_ids_from_advanced(self, soup: BeautifulSoup) -> Dict[str, str]:
        out: Dict[str, str] = {}
        table = self._table_from_soup(soup, "advanced-team")
        if table is None or table.tbody is None:
            return out
        for tr in table.tbody.find_all("tr"):
            if tr.get("class") and "thead" in tr.get("class", []):
                continue
            team_td = tr.find("td", {"data-stat": "team"})
            if team_td is None:
                continue
            a = team_td.find("a")
            if a is None:
                continue
            team = a.get_text(strip=True)
            href = a.get("href", "")
            m = re.search(r"/teams/([A-Z]{3})/", href)
            if m:
                out[team] = m.group(1)
        return out

    def _advanced_map(self, soup: BeautifulSoup) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {}
        table = self._table_from_soup(soup, "advanced-team")
        if table is None or table.tbody is None:
            return out
        for tr in table.tbody.find_all("tr"):
            if tr.get("class") and "thead" in tr.get("class", []):
                continue
            team_td = tr.find("td", {"data-stat": "team"})
            if team_td is None:
                continue
            team = team_td.get_text(strip=True).replace("*", "")
            out[team] = {
                "off_rtg": self._to_float((tr.find("td", {"data-stat": "off_rtg"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "off_rtg"}) else None),
                "def_rtg": self._to_float((tr.find("td", {"data-stat": "def_rtg"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "def_rtg"}) else None),
                "net_rtg": self._to_float((tr.find("td", {"data-stat": "net_rtg"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "net_rtg"}) else None),
                "pace": self._to_float((tr.find("td", {"data-stat": "pace"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "pace"}) else None),
            }
        return out

    def _team_splits(self, abbr: str, season: int) -> Dict[str, Optional[float]]:
        soup = self._get_soup(f"/teams/{abbr}/{season}/splits/")
        table = self._table_from_soup(soup, "team_splits")
        if table is None:
            return {
                "home_win_pct": None,
                "away_win_pct": None,
                "points_home": None,
                "points_away": None,
                "q4_points_for": None,
                "q4_points_against": None,
                "q4_net": None,
                "red_zone_conversion_proxy": None,
            }
        rows = []
        for tr in table.find_all("tr"):
            sid = tr.find("th", {"data-stat": "split_id"})
            sval = tr.find("td", {"data-stat": "split_value"})
            if sid is None or sval is None:
                continue
            rows.append(
                {
                    "split_id": sid.get_text(strip=True),
                    "split_value": sval.get_text(strip=True),
                    "wins": self._to_int((tr.find("td", {"data-stat": "wins"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "wins"}) else None),
                    "losses": self._to_int((tr.find("td", {"data-stat": "losses"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "losses"}) else None),
                    "pts": self._to_float((tr.find("td", {"data-stat": "pts"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "pts"}) else None),
                    "opp_pts": self._to_float((tr.find("td", {"data-stat": "opp_pts"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "opp_pts"}) else None),
                    "fg": self._to_float((tr.find("td", {"data-stat": "fg"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "fg"}) else None),
                    "fga": self._to_float((tr.find("td", {"data-stat": "fga"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "fga"}) else None),
                    "fg3": self._to_float((tr.find("td", {"data-stat": "fg3"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "fg3"}) else None),
                    "fg3a": self._to_float((tr.find("td", {"data-stat": "fg3a"}) or {}).get_text(strip=True) if tr.find("td", {"data-stat": "fg3a"}) else None),
                }
            )

        def find_row(split_id: str, split_value: str):
            for r in rows:
                if r["split_id"] == split_id and r["split_value"] == split_value:
                    return r
            return None

        home = find_row("Place", "Home")
        road = find_row("", "Road")
        total = find_row("", "Total")

        home_win_pct = None
        away_win_pct = None
        if home and home["wins"] is not None and home["losses"] is not None and (home["wins"] + home["losses"]) > 0:
            home_win_pct = home["wins"] / (home["wins"] + home["losses"])
        if road and road["wins"] is not None and road["losses"] is not None and (road["wins"] + road["losses"]) > 0:
            away_win_pct = road["wins"] / (road["wins"] + road["losses"])

        red_zone_proxy = None
        if total and None not in {total["fg"], total["fga"], total["fg3"], total["fg3a"]}:
            two_pm = max(0.0, (total["fg"] or 0.0) - (total["fg3"] or 0.0))
            two_pa = max(0.0, (total["fga"] or 0.0) - (total["fg3a"] or 0.0))
            red_zone_proxy = (two_pm / two_pa) if two_pa > 0 else None

        # Basketball-Reference team splits does not expose direct team-level Q4 points.
        # Keep these as nullable until a quarter-source endpoint is added.
        return {
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "points_home": home["pts"] if home else None,
            "points_away": road["pts"] if road else None,
            "q4_points_for": None,
            "q4_points_against": None,
            "q4_net": None,
            "red_zone_conversion_proxy": red_zone_proxy,
        }

    def scrape_team_profiles(self, season: int) -> List[Dict]:
        league = self._get_soup(f"/leagues/NBA_{season}.html")
        team_ids = self._team_ids_from_advanced(league)
        rmap = self._advanced_map(league)

        rows: List[Dict] = []
        for team, abbr in sorted(team_ids.items()):
            splits = self._team_splits(abbr, season)
            r = rmap.get(team, {})
            row = TeamProfile(
                team=team,
                team_abbr=abbr,
                season=season,
                off_rtg=r.get("off_rtg"),
                def_rtg=r.get("def_rtg"),
                net_rtg=r.get("net_rtg"),
                pace=r.get("pace"),
                home_win_pct=splits["home_win_pct"],
                away_win_pct=splits["away_win_pct"],
                points_home=splits["points_home"],
                points_away=splits["points_away"],
                q4_points_for=splits["q4_points_for"],
                q4_points_against=splits["q4_points_against"],
                q4_net=splits["q4_net"],
                red_zone_conversion_proxy=splits["red_zone_conversion_proxy"],
            )
            rows.append(asdict(row))
        return rows
