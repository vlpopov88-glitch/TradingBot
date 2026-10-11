# TradingBot: XAU/USD, Forex & Crypto

An automated trading bot for gold, major FX pairs and crypto. It analyzes markets, news and other traders' positioning around the clock, and trades with strict risk limits. It runs in **paper mode (simulated money) by default.**

> ⚠️ **Read this first.** No bot guarantees profit, and most retail algorithmic traders lose money. Every default in this bot was tested on years of real data (see [docs/RESEARCH.md](docs/RESEARCH.md)). The edge it found is **small**: about +4% per market over 5 years out of sample. Paper trade for weeks before considering real money, and only ever risk money you can afford to lose. This is software, not financial advice.

## Накратко на български

**Какво прави ботът:** На всеки 5 минути проверява злато (XAU/USD), EUR/USD, GBP/USD, USD/JPY, BTC и ETH.
- Следи тенденцията на дневни свещи.
- Чете новините (с Claude, ако има API ключ).
- Гледа позициите на големите спекуланти от отчетите на CFTC и залага *срещу* тяхното трупане, защото историческите данни показват, че те влизат късно.
- Спира новите сделки около важни икономически новини.
- Всяка сделка рискува най-много 1% от сметката.

**Как да го пуснете 24/7 на сървър (VPS):**
```bash
git clone https://github.com/vlpopov88-glitch/TradingBot.git && cd TradingBot
cp .env.example .env          # по желание: Telegram и API ключове
docker compose up -d --build  # стартира и рестартира автоматично
docker compose logs -f        # какво прави в момента
docker compose run --rm tradingbot status   # печалба/загуба и сделки
```
**Известия на телефона:** създайте Telegram бот чрез @BotFather и попълнете `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID` в `.env` (подробно в `tradingbot/notify.py`). Ще получавате съобщение при всяка сделка, при спиране на търговията и дневно обобщение.

**След 2–4 седмици:** пуснете `docker compose run --rm tradingbot evaluate`. Командата измерва кои сигнали (новини, позиции, техника) реално са предсказвали цената, за да коригираме тежестите им.

## What it does

1. **Technical analysis on daily bars:** trend (EMA 20/50/200), momentum (MACD + ROC) and Donchian breakout. Mean reversion is also available. Research showed that on hourly and 4-hour bars these markets mean-revert and every trend signal pointed the wrong way, so daily is the default.
2. **News:** RSS feeds (Investing.com, Cointelegraph, MarketWatch, Google News) are scored per asset and currency, with older headlines counting less. For a pair the score is `sentiment(base) − sentiment(quote)`.
   - **With `ANTHROPIC_API_KEY` set:** Claude reads the headlines and understands second-order effects (hot US CPI → stronger USD → weaker gold). Each headline is scored once and cached.
   - **Otherwise:** a keyword lexicon. It works, but it is crude.
3. **Watching other traders:**
   - **Gold & FX:** CFTC Commitments of Traders, weekly. The bot **fades** large speculators' 4-week flow and their crowded extremes. Following their flow was reliably wrong over 10 years (IC −0.07, t −3.1).
   - **Crypto:** OKX top-trader/crowd ratios and funding. **Off by default**: the only ~100 days of history pointed the wrong way, which is too little data to trust either direction.
4. **Risk management:**
   - 1% of equity at risk per trade, with an ATR stop, a 2R target and a trailing stop.
   - Stops are evaluated bar by bar, never against an earlier low.
   - Leverage caps, plus position limits in total and per asset class.
   - A 2-bar cooldown after an exit.
   - A daily-loss limit (3%) and a max-drawdown halt (15%). Both survive restarts and are lifted with `tradingbot resume`.
   - No trading on stale prices (weekends, holidays, dead feeds).
   - **Economic-calendar blackout:** no new entries from 30 minutes before to 30 minutes after high-impact releases (NFP, CPI, FOMC…).
   - A kill-switch file.
5. **Execution:** paper by default. Live mode sends gold and FX to **OANDA** and crypto to any **CCXT** exchange.
6. **Self-measurement:** every signal is journaled once per bar. `tradingbot evaluate` scores each one against what the price did next.

## How a decision is made

```
technical (trend + momentum + breakout, daily)  ─┐ 0.5
trader positioning (fade COT speculator flow)   ─┼ 0.3 × confidence ──► score ∈ [-1, 1]
news sentiment (Claude or keywords)              ─┘ 0.2 × confidence
score ≥ +0.35 → BUY · score ≤ −0.35 → SELL · open trade & score flips past 0.10 → CLOSE
```

Weights, thresholds, strategies, risk limits, markets and timeframe are all set in `config.yaml`.

## Run it 24/7 (Docker)

On any small Linux server (1 GB RAM is plenty) with Docker installed:

```bash
cp .env.example .env               # optional: Telegram, Claude, broker keys
docker compose up -d --build       # starts now and after every reboot
docker compose logs -f             # live log
docker compose run --rm tradingbot status     # P&L and recent trades
docker compose exec tradingbot touch /app/state/STOP   # kill switch: no new entries
docker compose down                # stop (state is kept in a Docker volume)
```

The container runs as a non-root user and has a health check: it reports *unhealthy* if no cycle has completed for 20 minutes. If Docker Hub rate-limits your server, build from a mirror: `docker build --build-arg BASE=mirror.gcr.io/library/python:3.12-slim .`

Without Docker: `pip install -r requirements.txt`, then `python -m tradingbot run`.

## Phone alerts (Telegram)

1. In Telegram, message **@BotFather**, send `/newbot` and copy the token.
2. Send your new bot any message. Then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy the number after `"chat":{"id":`.
3. Put `TELEGRAM_BOT_TOKEN=…` and `TELEGRAM_CHAT_ID=…` in `.env`, then run `python -m tradingbot notify-test`.

You'll get a message for every opened and closed trade (with P&L), when trading halts, when cycles keep failing, and a daily summary at 21:00 UTC (configurable).

## Commands

```bash
python -m tradingbot run             # trading loop (paper by default)
python -m tradingbot analyze         # what the bot thinks of every market right now
python -m tradingbot news            # scored headlines
python -m tradingbot status          # P&L summary and recent trades
python -m tradingbot evaluate        # which live signals actually predicted prices
python -m tradingbot research        # measure every signal on years of history
python -m tradingbot backtest        # backtest the current config (daily bars + COT)
python -m tradingbot resume          # lift a loss/drawdown halt after review
python -m tradingbot notify-test     # check Telegram alerts
python -m tradingbot flatten         # close everything
```

## What the research found

Full tables and method are in [docs/RESEARCH.md](docs/RESEARCH.md). Every choice below was made on 2016–2021 data and then checked on the unseen 2021–2026 data.

| Setup (daily bars, 6 markets) | Avg. return per market, Oct 2021 – Oct 2026 | Markets up |
|---|---|---|
| Original design (4-strategy blend + follow speculators) | −1.3% | 2/6 |
| Trend strategies only | +2.4% | 4/6 |
| **Current default (trend + fade speculators)** | **+4.0%** | 4/6 |

On hourly and 4-hour bars, every technical variant lost money out of sample. Don't switch to them expecting more profit.

## Markets (default)

| Market | Data | Positioning | Live broker |
|---|---|---|---|
| XAU/USD | Yahoo `GC=F` | CFTC gold (088691) | OANDA `XAU_USD` |
| EUR/USD, GBP/USD, USD/JPY | Yahoo | CFTC currency futures | OANDA |
| BTC/USD, ETH/USD | OKX spot | off by default | CCXT (spot, long only) |

To add a market, add an entry under `instruments:`. Position sizing supports only pairs that are USD-quoted or USD-based.

## Going live (only after weeks of paper trading)

1. Put your keys in `.env`. API keys need **trading permission only, never withdrawal**.
2. **OANDA:** start with `OANDA_ENV=practice` (a free demo account). Stops and targets sit on OANDA's servers, so they protect you even if the bot is down.
3. **Crypto:** `pip install ccxt`, with `CCXT_SANDBOX=1` (testnet) first. On spot markets the bot goes long only and manages stops itself.
4. Set `mode: live` in `config.yaml` and run `python -m tradingbot run --confirm-live`. With Docker, change the compose `command` to `["run", "--confirm-live"]`.

## Project layout

```
tradingbot/
  data/market.py   Yahoo, OKX, OANDA and synthetic candle providers
  indicators.py    EMA, RSI, ATR, ADX, MACD, Bollinger, Donchian
  strategies.py    strategies (per-bar score series) + regime-aware ensemble
  news.py          RSS ingestion + keyword sentiment
  llm_news.py      Claude headline scoring (optional)
  positioning.py   CFTC COT + OKX crypto positioning
  econ_calendar.py high-impact event blackout
  risk.py          sizing, stops, trailing, limits, persistent circuit breakers
  brokers/         paper, OANDA, CCXT, router
  engine.py        the live loop
  backtest.py      bar-by-bar backtester (supports dated context signals)
  research.py      signal research and live-signal evaluation
  notify.py        Telegram alerts
  journal.py       SQLite journal (trades, equity, per-bar signal snapshots)
docs/RESEARCH.md   what was tested and what was found
tests/             pytest suite: pip install -r requirements-dev.txt && python -m pytest
```
