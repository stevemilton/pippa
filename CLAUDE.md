# QuantAgent — Project Context

## What This Is
Personal algorithmic trading scanner (Phase 1 = scan only, no execution). Connects to Interactive Brokers via IB Gateway on a Hetzner VPS. Uses Student-t vine copula analysis to find joint probability mispricings across correlated instrument clusters. Claude API (claude-sonnet-4-6) is the orchestrating brain via tool_use.

## Architecture
```
IB Gateway (Docker) → MarketDataClient → CopulaEngine → Claude Agent → Telegram + SQLite
```

### Instrument Clusters
- **uk_macro**: FTSE100, GILT, GBP/USD, SHORT_STERLING, SONIA
- **us_rates**: ES, ZN, ZB, EUR/USD

### Scan Loop
Every 5 minutes: fetch returns → fit vine copula → compute implied vs copula probs → flag discrepancies > 3% → Claude analyses → Telegram alert + SQLite log.

## CRITICAL MATHEMATICAL REQUIREMENTS — NON-NEGOTIABLE

These are absolute. Do not weaken, approximate, or replace any of them:

1. **Student-t copula with ν=4 ONLY**. Gaussian copula is EXCLUDED from the family set. Gaussian sets tail dependence to zero — t(ν=4) gives ~18% tail dependence at ρ=0.5. This is where the edge lives. Formula: `λ_L = 2·t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))`

2. **Importance sampling for tails**. Any instrument priced <5% probability auto-switches to `rare_event_IS()` with exponential tilting. Crude MC on tails is useless.

3. **Variance reduction**. Antithetic variates + stratified sampling via `stratified_binary_mc()`. Target 100-500x over crude MC.

4. **Brier score calibration**. Every finding = a prediction. Track outcomes. Target Brier < 0.10. Weekly Telegram summary (Sunday 20:00 UTC).

5. **Pre-trade ABM check** (Phase 2 stub). Kyle's lambda market impact estimation before any order.

## Key Files

| File | Purpose |
|------|---------|
| `quantagent/copula.py` | Core engine — vine fitting, tail dependence, IS, variance reduction |
| `quantagent/math_models.py` | User-provided math implementations — DO NOT REWRITE |
| `quantagent/instruments.py` | Cluster definitions + futures auto-rollover |
| `quantagent/market_data.py` | IB Gateway connection via ib_insync |
| `quantagent/database.py` | SQLite (WAL mode) — scans, findings, predictions, brier_scores, errors |
| `quantagent/tools.py` | 9 tool definitions + handlers for Claude tool_use |
| `quantagent/agent.py` | Claude agent loop (max 10 tool rounds per scan) |
| `quantagent/notifications.py` | Telegram via httpx |
| `quantagent/main.py` | Scanner class, async scan loop, signal handling |
| `quantagent/outcomes.py` | Brier outcome recording CLI + auto-resolver |
| `quantagent/phase2/` | Stubs: execution, Kelly criterion, particle filter, Betfair, PostgreSQL |

## Running Tests
```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest tests/ -v
```
89 tests. Requires: numpy, scipy, pyvinecopulib, ib_insync, httpx, anthropic, python-dotenv, pytest, pytest-asyncio.

## Paper-Mode Validation
```bash
python scripts/validate_paper_mode.py --skip-telegram
```
10-step end-to-end check. Requires live IB Gateway connection (paper account).

## Deployment
- Hetzner VPS with `scripts/setup.sh`
- IB Gateway via `docker-compose.yml` (ports 4001/4002/5900)
- systemd service: `systemd/quantagent.service`
- Sunday 22:00 UTC: IB Gateway auto-restart via `scripts/ib-gateway-restart.sh`

## Phase 2 (Not Yet Built)
- `place_order` — real execution via IB, guarded by Kelly criterion position sizing
- Particle filter for latent probability tracking
- ABM market impact estimation (Kyle's lambda) as pre-trade check
- Betfair integration for prediction market cross-referencing
- PostgreSQL migration for production data

## pyvinecopulib API (v0.7.5+)
The API changed from earlier versions:
- `Vinecop(data, controls=...)` → `Vinecop.from_data(data, controls=...)`
- `vine.get_all_pair_copulas()` → `vine.families` (list of lists) or `vine.get_pair_copula(tree, edge)`

## Config
All via `.env` (see `.env.example`). Key vars: `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `IBKR_HOST`, `IBKR_PORT`, `PAPER_TRADING=true`.
