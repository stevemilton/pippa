#!/usr/bin/env bash
#
# Full setup script for a fresh Hetzner VPS.
# Run as root or with sudo.

set -euo pipefail

APP_DIR="/opt/quantagent"
APP_USER="quantagent"

echo "=== QuantAgent Setup ==="

# ── System packages ──────────────────────────────────────────────────────────
echo "Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    docker.io \
    docker-compose-plugin \
    build-essential \
    cmake \
    libboost-all-dev \
    git \
    curl \
    netcat-openbsd \
    sqlite3

# Enable Docker
systemctl enable docker
systemctl start docker

# ── App user ─────────────────────────────────────────────────────────────────
echo "Creating app user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -m -d "$APP_DIR" -s /bin/bash "$APP_USER"
fi
usermod -aG docker "$APP_USER"

# ── App directory ────────────────────────────────────────────────────────────
echo "Setting up app directory..."
mkdir -p "$APP_DIR"

# Copy project files (assumes you've cloned/copied them to /tmp/quantagent)
if [ -d "/tmp/quantagent" ]; then
    cp -r /tmp/quantagent/* "$APP_DIR/"
fi

# ── Python virtual environment ───────────────────────────────────────────────
echo "Creating Python virtual environment..."
python3.11 -m venv "$APP_DIR/venv"
source "$APP_DIR/venv/bin/activate"

echo "Installing Python dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r "$APP_DIR/requirements.txt"

# ── .env file ────────────────────────────────────────────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Creating .env from template..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "!! IMPORTANT: Edit $APP_DIR/.env with your credentials !!"
fi

# ── Permissions ──────────────────────────────────────────────────────────────
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 "$APP_DIR/.env"
chmod +x "$APP_DIR/scripts/"*.sh

# ── Systemd service ─────────────────────────────────────────────────────────
echo "Installing systemd service..."
cp "$APP_DIR/systemd/quantagent.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable quantagent

# ── Cron for Sunday restart ─────────────────────────────────────────────────
echo "Installing IB Gateway restart cron..."
CRON_LINE="0 22 * * 0 $APP_DIR/scripts/ib-gateway-restart.sh >> /var/log/ib-gateway-restart.log 2>&1"
(crontab -u "$APP_USER" -l 2>/dev/null | grep -v "ib-gateway-restart"; echo "$CRON_LINE") | crontab -u "$APP_USER" -

# ── Docker compose ───────────────────────────────────────────────────────────
echo "Starting IB Gateway via Docker..."
cd "$APP_DIR"
sudo -u "$APP_USER" docker compose up -d

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $APP_DIR/.env with your IBKR, Telegram, and Anthropic credentials"
echo "  2. Connect to VNC (port 5900) to complete IB Gateway initial login"
echo "  3. Start the scanner: systemctl start quantagent"
echo "  4. Check logs: journalctl -u quantagent -f"
echo ""
