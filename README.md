# Double Bottom Scanner v5.2

Production-grade double bottom pattern scanner with optimised parameters derived
from a 5-year, 105-ticker backtest sweep.

## Backtest metrics
| Metric | Value |
|--------|-------|
| CAGR | 11.17% |
| Sharpe | 1.858 |
| Calmar | 1.27 |
| Max Drawdown | $-1,760 |
| Win Rate | 29.3% (BE: 22.2%) |

## Config
| Parameter | Value |
|-----------|-------|
| TOL | 0.02 |
| Trend filter | SMA50 |
| Entry | Open > neckline (confirmation) |
| SL | 2% |
| TP | 7% |
| MaxHold | 30 sessions |
| Notional | $2,000/trade |
| Max positions | 10 |

## Setup

```bash
pip install -r requirements.txt
```

Add your tickers to `tickers.txt` (one per line).

## Run locally

```bash
# Streamlit app
streamlit run app.py

# CLI only
python double_bottom_scanner_v5_2.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to share.streamlit.io → New app
3. Select repo + set main file to `app.py`
4. Deploy

> Research only. Not financial advice.
