from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from .config import BotSettings
from .data_rest import KalshiDataClient
from .watchlist import build_watchlist, watchlist_as_dict


HTML_PAGE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Kalshi Watchlist</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 8px; }
      th { background: #f4f4f4; }
      .lane { text-transform: uppercase; font-weight: bold; }
    </style>
  </head>
  <body>
    <h2>Kalshi Watchlist</h2>
    <div id="meta"></div>
    <table id="table">
      <thead>
        <tr>
          <th>Lane</th>
          <th>Ticker</th>
          <th>Title</th>
          <th>Status</th>
          <th>Spread</th>
          <th>Trades(1h)</th>
          <th>Trades(60m)</th>
          <th>DepthTop3</th>
          <th>Close</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
    <script>
      async function refresh() {
        const res = await fetch('/watchlist.json');
        const data = await res.json();
        const tbody = document.querySelector('#table tbody');
        tbody.innerHTML = '';
        document.getElementById('meta').textContent = 'Updated: ' + data.updated_at;
        for (const item of data.items) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="lane">${item.lane}</td>
            <td>${item.ticker}</td>
            <td>${item.title || ''}</td>
            <td>${item.status || ''}</td>
            <td>${item.spread_cents ?? ''}</td>
            <td>${item.trades_1h ?? ''}</td>
            <td>${item.trades_60m ?? ''}</td>
            <td>${item.depth_top3 ?? ''}</td>
            <td>${item.close_time ?? ''}</td>
          `;
          tbody.appendChild(tr);
        }
      }
      refresh();
      setInterval(refresh, 10000);
    </script>
  </body>
</html>
"""


class WatchlistHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return
        if self.path.startswith("/watchlist.json"):
            payload = self.server.watchlist_payload()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        return


class WatchlistHTTPServer(HTTPServer):
    def __init__(self, addr, handler, payload_ref):
        super().__init__(addr, handler)
        self._payload_ref = payload_ref

    def watchlist_payload(self):
        return self._payload_ref["payload"]


def serve_watchlist(
    settings: BotSettings,
    data_client: KalshiDataClient,
    host: str,
    port: int,
    top: int,
    include_weather: bool,
    include_sports: bool,
    refresh_sec: int,
) -> None:
    payload_ref = {"payload": {"updated_at": None, "items": []}}

    def refresher():
        while True:
            items = build_watchlist(
                settings,
                data_client,
                top=top,
                include_weather=include_weather,
                include_sports=include_sports,
            )
            payload_ref["payload"] = watchlist_as_dict(items)
            time.sleep(refresh_sec)

    thread = threading.Thread(target=refresher, daemon=True)
    thread.start()

    server = WatchlistHTTPServer((host, port), WatchlistHandler, payload_ref)
    server.serve_forever()
