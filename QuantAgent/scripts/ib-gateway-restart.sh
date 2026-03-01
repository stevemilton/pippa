#!/usr/bin/env bash
#
# Sunday night IB Gateway restart script.
# IBKR requires a daily restart; this does it cleanly on Sunday night
# before the Asian session opens (futures open Sunday ~6pm ET / 11pm UTC).
#
# Install as a cron job:
#   0 22 * * 0 /opt/quantagent/scripts/ib-gateway-restart.sh >> /var/log/ib-gateway-restart.log 2>&1

set -euo pipefail

COMPOSE_DIR="/opt/quantagent"
LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"

echo "$LOG_PREFIX Starting IB Gateway restart..."

cd "$COMPOSE_DIR"

# Graceful stop
echo "$LOG_PREFIX Stopping IB Gateway container..."
docker compose stop ib-gateway

# Wait for clean shutdown
sleep 10

# Remove old container (keeps volume)
docker compose rm -f ib-gateway

# Pull latest image (optional — comment out if you want to pin versions)
# docker compose pull ib-gateway

# Start fresh
echo "$LOG_PREFIX Starting IB Gateway container..."
docker compose up -d ib-gateway

# Wait for gateway to be ready
echo "$LOG_PREFIX Waiting for IB Gateway to become healthy..."
for i in $(seq 1 30); do
    if docker compose exec -T ib-gateway nc -z localhost 4002 2>/dev/null; then
        echo "$LOG_PREFIX IB Gateway is healthy after ${i}0 seconds"
        exit 0
    fi
    sleep 10
done

echo "$LOG_PREFIX WARNING: IB Gateway did not become healthy within 5 minutes"
exit 1
