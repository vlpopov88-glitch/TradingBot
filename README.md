# TradingBot — XAU/USD, Forex & Crypto

An automated trading bot that continuously:

1. **Runs technical strategies**: trend following (EMA 20/50/200), momentum (MACD + ROC), Donchian breakout, and mean reversion (RSI + Bollinger). It blends them according to the market regime detected by ADX: trend strategies get more weight in trending markets and mean reversion more in ranging ones.
2. **Analyzes news**: it polls RSS feeds (Investing.com, Cointelegraph, MarketWatch, Google News) and scores every headline per asset and currency. Older headlines count less (exponential decay). For a pair, the score is `sentiment(base) − sentiment(quote)`.
   - **With `ANTHROPIC_API_KEY` set** (and `pip install anthropic`): Claude reads the headlines in batches and returns the expected direction for each affected asset, including second-order effects such as hot US CPI → stronger USD → weaker gold. Each headline is scored once and cached, so steady-state cost is only the few new headlines per refresh. The model and effort level are set in `config.yaml` under `news.llm`.
   - **Without it, or if a call fails:** a keyword lexicon covering hawkish/dovish central-bank tone and safe-haven flows. It works, but misreads context (e.g. "Bitcoin stuck in range").
3. **Watches other traders**:
   - **Gold & forex:** CFTC Commitments of Traders. The bot ranks large speculators' net position against the last 3 years. It follows their flows in the mid-range and fades them at crowded extremes.
   - **Crypto:** OKX top-trader long/short ratio (follow the "smart money"), all-accounts ratio (fade the retail crowd) and the funding rate (fade over-leveraged positioning).
4. **Manages risk**:
   - Each trade risks 1% of equity, with an ATR-based stop, a 2R take-profit and a trailing stop.
   - Leverage caps per asset class, plus limits on open positions in total and per class.
   - A daily-loss circuit breaker and a max-drawdown halt.
   - No trading on stale data (weekends and holidays).
   - **Economic-calendar blackout:** no new entries 30 minutes before to 30 minutes after high-impact releases (NFP, CPI, FOMC, ...) for the affected currencies. Gold and crypto count as USD-exposed. Events come from the Forex Factory weekly feed.
   - A kill switch file.
5. **Executes**: in paper mode by default. In live mode it routes gold and FX to **OANDA** and crypto to any **CCXT** exchange.

Every decision, trade and equity point goes into a SQLite journal (`state/journal.db`).

> ⚠️ **Read this first.** No bot guarantees profit, and most retail algorithmic traders lose money. The backtests below show the technical layer **does not have a reliable edge as shipped**. Run it in paper mode for weeks, study the journal and improve it before you put real money in. Trade only money you can afford to lose. This is software, not financial advice.

## Quick start

```bash
pip install -r requirements.txt

python -m tradingbot analyze      # what the bot thinks of every market right now
python -m tradingbot news         # scored headlines
python -m tradingbot backtest     # test the technical strategies on real history
python -m tradingbot run          # start paper trading (loops every 5 min)
python -m tradingbot status       # P&L summary and recent trades
python -m tradingbot flatten      # close everything
```

To block new entries immediately, run `touch state/STOP`. Delete the file to resume.

Example `analyze` output:

```
XAU/USD @ 4216.3  ->  HOLD  (score +0.11)
   technical       +0.16  conf 1.00  trending market (ADX 40)
   news            +0.09  conf 0.69  3.5 weighted headlines; e.g. 'Gold Price Outlook: ...'
   positioning     -0.19  conf 0.60  COT index 86%, net +53.1% of OI, 4wk chg -3.3%
   trend           +0.68  ...
```

## How a decision is made

```
technical (regime-weighted blend of 4 strategies) ──┐ 0.60
news sentiment (decayed, per currency/asset)  ──────┼ 0.25 × confidence ──► score ∈ [-1, 1]
trader positioning (COT / OKX)               ───────┘ 0.15 × confidence
score ≥ +0.35 → BUY · score ≤ −0.35 → SELL · open trade & score flips past 0.10 → CLOSE
```

Everything above is configurable in `config.yaml`: weights, thresholds, risk limits, markets and timeframe.

## Markets (default)

| Market  | Data                | Positioning           | Live broker |
|---------|---------------------|-----------------------|-------------|
| XAU/USD | Yahoo `GC=F`        | CFTC gold (088691)    | OANDA `XAU_USD` |
| EUR/USD, GBP/USD, USD/JPY | Yahoo | CFTC currency futures | OANDA |
| BTC/USD, ETH/USD | OKX spot   | OKX top traders, crowd, funding | CCXT (spot, long only) |

To add a market, add an entry under `instruments:`. Only USD-quoted or USD-based pairs are supported for position sizing.

## Going live (only after paper trading)

1. Copy `.env.example` to `.env`, fill in your keys and export them.
2. **OANDA:** start with `OANDA_ENV=practice` (a free demo account). Stops and targets are attached server-side, so they protect you even if the bot crashes.
3. **Crypto:** run `pip install ccxt`. Set `CCXT_SANDBOX=1` (testnet) first. On spot markets the bot only goes long, and the bot itself manages the stops.
4. Set `mode: live` in `config.yaml` and run `python -m tradingbot run --confirm-live`.

The API keys you create should have **trading permission only, never withdrawal permission**.

## Backtest results (honest)

Real data, technical layer only (no news or positioning history), 1% risk per trade, with spread costs:

| Market | 1h bars (~4 months) | 4h bars (~1 year) |
|--------|--------------------|-------------------|
| XAU/USD | −5.1% | +7.3% |
| EUR/USD | −7.7% | +3.7% |
| GBP/USD | −4.0% | −6.2% |
| USD/JPY | −9.0% | −5.9% |
| BTC/USD | −3.3% | −2.7% |
| ETH/USD | +6.2% | −7.7% |

Conclusion: textbook indicators alone are not an edge. Don't tune parameters until these numbers turn green. That's overfitting, and it falls apart live. Better next steps:
- Collect a few weeks of journal data. Measure whether the news and positioning signals actually predict returns, then adjust the weights.
- Walk-forward / out-of-sample testing for any change.

## Project layout

```
tradingbot/
  data/market.py   Yahoo, OKX, OANDA and synthetic candle providers
  indicators.py    EMA, RSI, ATR, ADX, MACD, Bollinger, Donchian
  strategies.py    4 strategies + regime-aware ensemble
  news.py          RSS ingestion + keyword sentiment
  llm_news.py      Claude headline scoring (optional)
  econ_calendar.py high-impact event blackout
  positioning.py   CFTC COT + OKX crypto trader positioning
  risk.py          sizing, stops, trailing, limits, circuit breakers
  brokers/         paper, OANDA, CCXT, router
  engine.py        the live loop
  backtest.py      bar-by-bar backtester
  journal.py       SQLite journal
tests/             pytest suite (python -m pytest)
```
