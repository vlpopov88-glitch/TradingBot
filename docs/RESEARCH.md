# Signal research: what actually predicts prices?

*Run on 2026-10-11 with `python -m tradingbot research` (re-run it any time; numbers will drift as new data arrives).*

The bot's defaults come from this research, not from intuition. Each signal's score
at decision time was lined up with the price move that followed. The report has these columns:

- **IC** is the rank correlation between score and forward return. 0 means useless. In liquid markets a genuine edge is usually small, around 0.02–0.05.
- **t** is the t-statistic of IC, counting overlapping horizons once. Anything below about 2 in absolute value can be luck. Markets are pooled, and EUR/GBP and BTC/ETH move together, so pooled t-values flatter the evidence.
- **1st / 2nd** are the IC in each half of the sample. A real effect keeps its sign in both halves.

Data used:
- Yahoo prices: 2 years of hourly bars and 10 years of daily bars for gold, EUR/USD, GBP/USD, USD/JPY, BTC and ETH.
- CFTC Commitments of Traders: 10 years of weekly reports.
- OKX crypto positioning: about 100–180 days, all that is publicly available.

## 1. Technical strategies

| Timeframe | Trend / momentum / breakout | Mean reversion |
|---|---|---|
| 1h bars | **Inverted**: breakout IC −0.025 (t −6.7), momentum −0.018 (t −4.7), trend −0.010 | IC +0.024 (t +6.2), both halves positive |
| 4h bars | Inverted or none | IC +0.018 (t +2.3) |
| 1d bars | No significant edge; trend +0.037 at 20 days (t 1.0) | None |

On hourly and 4-hour bars these markets mean-revert: breakouts fail and momentum fades.
Mean reversion is statistically real, but **following its strong signals lost money once
spreads and fees were included** (gross edge of a few basis points, smaller than costs).

### Out-of-sample backtest

Each setup was backtested on the first half of the history. The choice was then checked on the unseen second half, with spread and fee costs included (FX 1 bp, gold 2 bp, crypto 10 bp per fill). Results are averages over the 6 markets.

| Setup | 1h (2025–26) | 4h (2025–26) | 1d (Oct 2021 – Oct 2026) |
|---|---|---|---|
| Old 4-strategy blend | −8.6% | −3.9% | −1.1% |
| Mean reversion only | −7.4% | −4.2% | −0.8% |
| **Trend set (trend + momentum + breakout)** | −8.5% | −5.0% | **+2.4%** (4/6 markets up) |

Only daily trend-following stayed positive out of sample, which is consistent with decades of
published research on time-series momentum in futures. The edge is small.

## 2. Positioning ("watching other traders")

**CFTC COT, gold + FX, weekly.** The original rule bought when large speculators had been adding longs over the past 4 weeks.
Following that flow was **reliably wrong**: IC −0.067 at 1 week (t −3.1), negative in both halves and in 3 of 4
markets. Speculators tend to pile in late. Fading the extremes alone had too few cases to tell.

So the signal now **fades the speculators' 4-week flow** and fades crowded extremes. The new version scores IC +0.038 (1 week),
+0.039 (2 weeks) and +0.055 (4 weeks), positive in both halves. That is weak but consistent evidence.

**OKX crypto data (top traders, crowd, funding).** Over the roughly 100 days available, the "follow top traders /
fade the crowd" composite pointed the wrong way (IC −0.33 at 1 day for both BTC and ETH). That sample is far too short
to justify trading the reverse, so crypto positioning is **off by default** (`positioning.crypto: false`).

## 3. Final comparison on daily bars (Oct 2021 – Oct 2026)

Everything here was chosen using the first half (2016–2021). The COT flip was also visible in the first half: IC −0.045 there.

| Setup | Avg. return per market | Markets up | Avg. Sharpe | Worst drawdown |
|---|---|---|---|---|
| Old default (4-strategy blend + follow-COT) | −1.3% | 2/6 | −0.11 | −8.0% |
| Trend set only | +2.4% | 4/6 | 0.19 | −9.0% |
| **New default (trend set + fade-COT)** | **+4.0%** | 4/6 | 0.31 | −8.1% |

Per market, new default: gold +13.6%, ETH +6.3%, USD/JPY +5.6%, BTC +4.7%, GBP/USD −2.9%, EUR/USD −3.7%.

## 4. What this means

- The new defaults are **better than the original design, but the edge is small**: about 4% over 5 years per market at 1% risk per trade. That is not a money machine, and past results may not persist.
- News sentiment has **no free history**, so it could not be tested here. The bot records every signal once per bar while it runs. After a few weeks of paper trading, `python -m tradingbot evaluate` measures news, positioning and technical signals on what really happened, and the weights can then be adjusted on evidence.
- Do not switch to hourly bars expecting more profit. The research says the opposite.
