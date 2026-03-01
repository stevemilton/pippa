# QuantAgent

Personal algorithmic trading scanner. Connects to Interactive Brokers via IB Gateway, uses Student-t vine copula analysis (nu=4, Gaussian excluded) to detect correlated market mispricings, and sends Telegram alerts with Claude's reasoning.

**Phase 1: Scanner and validation only.** No trade execution.

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Main Loop                   │
│  (scan every N minutes across all clusters)  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│             Claude Agent (brain)             │
│  Calls tools, interprets results, reasons   │
└──┬────────┬────────┬────────┬────────┬──────┘
   │        │        │        │        │
   ▼        ▼        ▼        ▼        ▼
Market   Copula   Positions  SQLite  Telegram
 Data    Engine     (IBKR)    Log     Notify
(IBKR)  (t-vine)
```

## Mathematical Requirements (Non-Negotiable)

These choices are where the alpha lives. Do not substitute simpler alternatives.

### Student-t Copula (nu=4)

The Gaussian copula is **explicitly excluded** from the family set. It sets tail dependence to zero and systematically underestimates joint extreme outcomes by 2-5x. The Student-t copula with nu=4 gives approximately 18% tail dependence on correlated instruments:

```
lambda_L = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
```

For rho=0.5, nu=4: lambda_L ~ 0.18. Gaussian: lambda_L = 0. That gap is the edge.

### Mispricing Detection

For each instrument cluster:
1. Compute what each market implies independently (marginal probabilities)
2. Compute what the joint t-copula model says those probabilities should be given the dependency structure
3. The mispricing score is the divergence between these
4. Flag anything above `MISPRICING_THRESHOLD` (default 0.03)

### Automatic Importance Sampling for Tail Events

When any instrument in a cluster is priced below 5% probability, the system automatically switches from standard Monte Carlo to importance sampling with exponential tilting (`rare_event_IS`). Crude Monte Carlo on tail contracts gives estimates with useless precision. This switch is automatic and non-optional.

### Variance Reduction

Antithetic variates and stratified sampling (`stratified_binary_mc`) are applied alongside every copula analysis. Target: 100-500x variance reduction over crude MC. The variance reduction factor is reported in every finding.

### Brier Score Calibration

Every mispricing flagged is a prediction. When the outcome is known, it is recorded. Running Brier score is computed per instrument cluster. This is the **only honest measure** of whether the model has edge. It is surfaced in every weekly Telegram summary. Target: below 0.10.

### Pre-Trade ABM Impact Check (Phase 2)

Stubbed: takes proposed position size and uses `PredictionMarketABM` to estimate market impact before execution. This prevents moving the price against ourselves on thin markets.

## Repo Structure

```
QuantAgent/
├── .env.example              # Config template
├── docker-compose.yml        # IB Gateway container
├── requirements.txt
├── quantagent/
│   ├── main.py               # Entry point + scan loop + weekly summary
│   ├── config.py             # .env loading
│   ├── agent.py              # Claude tool_use orchestration
│   ├── tools.py              # Tool definitions + dispatch
│   ├── market_data.py        # IBKR via ib_insync
│   ├── copula.py             # Student-t vine copula engine (nu=4)
│   ├── instruments.py        # Cluster definitions
│   ├── notifications.py      # Telegram (alerts + weekly Brier summary)
│   ├── database.py           # SQLite logging + Brier score tracking
│   ├── math_models.py        # Monte Carlo, IS, particle filter, ABM
│   └── phase2/               # Stubbed for future
│       ├── execution.py      # place_order + ABM market impact
│       ├── position_sizing.py # Kelly criterion
│       ├── particle_filter.py # Bootstrap PF
│       ├── betfair.py        # Betfair data source
│       └── postgres.py       # PostgreSQL migration
├── scripts/
│   ├── setup.sh              # Full VPS setup
│   └── ib-gateway-restart.sh # Sunday night restart
└── systemd/
    └── quantagent.service    # Auto-start + crash recovery
```

## Hetzner Deployment Guide

### 1. Provision the VPS

- Hetzner Cloud: CX22 or better (2 vCPU, 4GB RAM)
- OS: Ubuntu 24.04
- Location: Falkenstein or Nuremberg (low latency to European exchanges)
- Enable backups

### 2. Initial Server Setup

```bash
ssh root@YOUR_SERVER_IP

# Basic security
apt update && apt upgrade -y
adduser deploy
usermod -aG sudo deploy

# Set up firewall
ufw allow 22/tcp
ufw allow 5900/tcp   # VNC for IB Gateway setup (remove after)
ufw enable
```

### 3. Clone and Run Setup

```bash
# As root
cd /tmp
git clone YOUR_REPO_URL quantagent
bash /tmp/quantagent/scripts/setup.sh
```

The setup script installs:
- Python 3.11 + venv
- Docker + Docker Compose plugin
- Build tools for pyvinecopulib (cmake, boost)
- Creates `quantagent` service user
- Sets up the virtualenv and installs all Python deps
- Installs the systemd service
- Sets up the Sunday night cron job
- Starts IB Gateway via Docker

### 4. Configure Credentials

```bash
sudo nano /opt/quantagent/.env
```

Fill in:
- `IBKR_USERNAME` / `IBKR_PASSWORD` — your Interactive Brokers credentials
- `IBKR_TRADING_MODE` — `paper` for testing, `live` for real
- `TELEGRAM_BOT_TOKEN` — create a bot via @BotFather on Telegram
- `TELEGRAM_CHAT_ID` — send a message to your bot, then hit `https://api.telegram.org/botYOUR_TOKEN/getUpdates` to find your chat ID
- `ANTHROPIC_API_KEY` — from console.anthropic.com

### 5. IB Gateway Initial Login

The first time IB Gateway starts, you need to complete the login via VNC:

```bash
# From your local machine
vncviewer YOUR_SERVER_IP:5900
```

- Password is whatever you set as `VNC_PASSWORD` in .env (default: `quantagent`)
- Complete the IB Gateway login prompt
- Accept any security prompts
- After successful login, the gateway will remember credentials for auto-login

**After initial setup, remove VNC from firewall:**
```bash
ufw delete allow 5900/tcp
```

### 6. Start QuantAgent

```bash
sudo systemctl start quantagent
sudo journalctl -u quantagent -f   # Watch logs
```

You should see:
- Connection to IB Gateway
- Startup Telegram notification (includes copula config)
- First scan cycle beginning

### 7. Verify It's Working

- Check Telegram for the startup notification
- Check logs: `journalctl -u quantagent --since "5 minutes ago"`
- Check the database: `sqlite3 /opt/quantagent/quantagent.db "SELECT * FROM scans ORDER BY id DESC LIMIT 5;"`
- Check Brier tracking: `sqlite3 /opt/quantagent/quantagent.db "SELECT * FROM predictions ORDER BY id DESC LIMIT 10;"`

### 8. Ongoing Operations

**Check status:**
```bash
systemctl status quantagent
```

**View recent logs:**
```bash
journalctl -u quantagent --since "1 hour ago"
```

**Query findings (with tail dependence and IS status):**
```bash
sqlite3 /opt/quantagent/quantagent.db \
  "SELECT timestamp, instruments, discrepancy, confidence, is_tail_regime, variance_reduction FROM findings ORDER BY timestamp DESC LIMIT 20;"
```

**Check Brier scores:**
```bash
sqlite3 /opt/quantagent/quantagent.db \
  "SELECT * FROM brier_scores ORDER BY timestamp DESC LIMIT 10;"
```

**Check unresolved predictions (need outcomes recorded):**
```bash
sqlite3 /opt/quantagent/quantagent.db \
  "SELECT * FROM predictions WHERE resolved = 0 ORDER BY timestamp ASC LIMIT 20;"
```

**Restart after config change:**
```bash
sudo systemctl restart quantagent
```

**IB Gateway auto-restarts** every Sunday at 22:00 UTC via cron. Weekly Brier summary sends at 20:00 UTC Sunday (before restart). Check restart logs:
```bash
cat /var/log/ib-gateway-restart.log
```

## Instrument Clusters

### uk_macro (default)
- FTSE 100 futures (ICE)
- UK Long Gilt futures (ICE)
- GBP/USD forex
- Short Sterling / SONIA futures (BoE rate expectations)

### us_rates
- E-mini S&P 500 (CME)
- 10-Year T-Note futures (CBOT)
- 30-Year T-Bond futures (CBOT)
- EUR/USD forex

Add new clusters in `quantagent/instruments.py`.

## How It Works

1. Every `SCAN_INTERVAL_MINUTES`, the scanner iterates over all active clusters.
2. For each cluster, Claude (via tool_use) is asked to run a scan.
3. Claude calls `get_market_data` to fetch live prices.
4. Claude calls `run_copula_analysis` which:
   - Fetches 60 days of historical returns
   - Transforms to pseudo-observations (rank to uniform)
   - Fits a Student-t vine copula (nu=4, Gaussian excluded)
   - Allowed families: Student-t, Clayton, Gumbel, Joe, Frank, BB1, BB7
   - Computes marginal-implied vs copula-conditional probabilities
   - Auto-switches to importance sampling for any instrument below 5%
   - Applies antithetic variates + stratified sampling (100-500x VR)
   - Computes pairwise tail dependence coefficients
   - Returns discrepancies with confidence scores
5. If max discrepancy exceeds the threshold, Claude:
   - Logs the finding to SQLite (auto-creates Brier prediction)
   - Sends a Telegram alert with full analysis including tail dependence and IS status
   - Explains in plain English why it matters
6. Every Sunday at 20:00 UTC, sends weekly summary with Brier scores per cluster.

## Phase 2 Hooks (Not Yet Implemented)

- `place_order` tool for automated execution
- `estimate_market_impact` — ABM pre-trade check to avoid moving price on thin markets
- Kelly criterion position sizing (full, fractional, portfolio)
- Bootstrap particle filter for live event tracking
- Betfair API as second data source
- PostgreSQL upgrade from SQLite
