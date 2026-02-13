#!/usr/bin/env bash
set -euo pipefail

KEY_PATH="/opt/render/project/src/kalshi_private_key.pem"
if [ -n "${KALSHI_PRIVATE_KEY_PEM:-}" ]; then
  printf "%s" "${KALSHI_PRIVATE_KEY_PEM}" > "$KEY_PATH"
  chmod 600 "$KEY_PATH"
  export KALSHI_PRIVATE_KEY_PATH="$KEY_PATH"
fi

python -m kalshi_bot.cli run-weather --live --i-understand-risk --cycles 0 --sleep 15
