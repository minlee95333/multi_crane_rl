#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m http.server 8000 --bind 0.0.0.0 >/tmp/crane_http.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
while true; do
  # localtunnel sometimes disconnects; restart it and keep the latest URL in /tmp/crane_public_url.txt
  npx --yes localtunnel --port 8000 2>&1 | tee /tmp/crane_lt.log | while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == your\ url\ is:* ]]; then
      url=${line#your url is: }
      echo "$url" > /tmp/crane_public_url.txt
      echo "PUBLIC_URL=$url"
    fi
  done
  echo "localtunnel disconnected; restarting in 2s"
  sleep 2
done
