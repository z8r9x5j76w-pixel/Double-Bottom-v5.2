#!/usr/bin/env python3
"""
DOUBLE BOTTOM SCANNER v5.2 — Production
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Config (locked from v5.1 optimisation):
  TOL       = 0.02   (tight double-bottom symmetry)
  SMA50     = on     (trend filter)
  Open conf = on     (entry only if open > neckline)
  SL        = 2%
  TP        = 7%
  MaxHold   = 30 sessions

Backtest metrics (5yr, 105-ticker universe):
  Sharpe 1.895 | CAGR 11.41% | Calmar 1.296 | MaxDD -$1,760

Output:
  1. TODAY'S ACTION SHEET  — entries for next open + exits due MaxHold
  2. OPEN POSITIONS        — live P&L snapshot
  3. BACKTEST SUMMARY      — headline metrics
  4. LATEST 10 CLOSED TRADES
  5. CSV export            — double_bottom_trades.csv
"""

import sys, time
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from pandas.tseries.offsets import BDay

# ── Config ────────────────────────────────────────────────────────────────────
PERIOD        = "5y"
INTERVAL      = "1d"
LOOKBACK      = 180
MIN_SEP       = 7
MAX_SEP       = 120
TOL           = 0.02
TREND_SMA     = 50
SL            = 0.02
TP            = 0.07
MAX_HOLD      = 30
NOTIONAL      = 2000.0
MAX_POSITIONS = 10
BATCH_SIZE    = 20
TRADING_DAYS  = 252


# ── Universe ──────────────────────────────────────────────────────────────────

def load_universe():
    p = Path("tickers.txt")
    if not p.exists():
        print("ERROR: tickers.txt not found in current folder."); raise SystemExit(1)
    tickers = []
    for line in p.read_text().splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#"):
            continue
        tickers.append(s.replace(".", "-"))
    return sorted(list(dict.fromkeys(tickers)))


# ── Download ──────────────────────────────────────────────────────────────────

def progress(current, total, prefix="", bar_len=38):
    pct    = current / total
    filled = int(bar_len * pct)
    bar    = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({pct*100:.0f}%)")
    sys.stdout.flush()
    if current == total:
        print()


def download_data(universe):
    batches = [universe[i:i+BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]
    data    = {}
    print(f"Downloading {len(universe)} tickers in {len(batches)} batches...")
    for bi, batch in enumerate(batches, 1):
        progress(bi - 1, len(batches), "Data    ")
        raw = yf.download(batch, period=PERIOD, interval=INTERVAL,
                          group_by="ticker", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            continue
        for t in batch:
            try:
                if hasattr(raw.columns, "levels") and len(raw.columns.levels) > 1:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t].copy()
                else:
                    df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.dropna()
                if not {"Open","High","Low","Close"}.issubset(df.columns):
                    continue
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df.sort_index()
                if len(df) < 250:
                    continue
                data[t] = df[["Open","High","Low","Close"]]
            except Exception:
                continue
    progress(len(batches), len(batches), "Data    ")
    return data


# ── Pattern detection ─────────────────────────────────────────────────────────

def pivot_lows_vec(arr, left=2, right=2):
    n   = len(arr)
    out = np.ones(n, dtype=bool)
    for offset in range(-left, right + 1):
        if offset == 0:
            continue
        i_s = max(0, -offset); i_e = min(n, n - offset)
        j_s = i_s + offset
        out[i_s:i_e] &= (arr[i_s:i_e] <= arr[j_s:j_s+(i_e-i_s)])
    out[:left]  = False
    out[n-right:] = False
    return out


def find_neckline(df, di):
    start   = max(0, di - LOOKBACK + 1)
    window  = df.iloc[start:di + 1]
    w_lows  = window["Low"].values.astype(float)
    w_highs = window["High"].values.astype(float)
    piv     = pivot_lows_vec(w_lows, 2, 2)
    idxs    = np.where(piv)[0]
    if len(idxs) < 2:
        return None
    best_score = -np.inf
    best_neck  = None
    for a in range(len(idxs) - 1):
        i = idxs[a]
        for b in range(a + 1, len(idxs)):
            j   = idxs[b]
            sep = j - i
            if sep < MIN_SEP: continue
            if sep > MAX_SEP: break
            low1, low2 = w_lows[i], w_lows[j]
            if low1 <= 0 or low2 <= 0: continue
            avg = (low1 + low2) / 2.0
            if abs(low1 - low2) / avg > TOL: continue
            neck = float(w_highs[i:j+1].max())
            if neck <= 0: continue
            score = j - (abs(low1 - low2) / avg) * 1000.0
            if score > best_score:
                best_score = score
                best_neck  = neck
    return best_neck


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(data, dates):
    """Full backtest with open-confirmation entry. Returns trades, positions, signals."""
    open_pos   = {}
    pending    = {}       # next_date -> [(ticker, strength, neckline)]
    trades     = []
    daily_pnl  = np.zeros(len(dates))
    n_dates    = len(dates)
    sma_cache  = {t: df["Close"].rolling(TREND_SMA).mean() for t, df in data.items()}

    for di in range(1, n_dates):
        date = dates[di]

        # ── Enter pending (open-confirmation) ─────────────────────────────────
        if date in pending:
            cands = sorted(pending.pop(date), key=lambda x: x[1], reverse=True)
            for ticker, strength, neckline in cands:
                if len(open_pos) >= MAX_POSITIONS: break
                if ticker in open_pos: continue
                df = data.get(ticker)
                if df is None or date not in df.index: continue
                o  = float(df.loc[date, "Open"])
                if not np.isfinite(o) or o <= 0: continue
                if o <= neckline: continue          # open-confirmation filter
                shares = NOTIONAL / o
                open_pos[ticker] = dict(
                    entry_date  = date,
                    entry_price = o,
                    shares      = shares,
                    sl_price    = o * (1 - SL),
                    tp_price    = o * (1 + TP),
                    neckline    = neckline,
                    days_held   = 0,
                    entry_di    = di,
                )

        # ── Exit logic ────────────────────────────────────────────────────────
        day_pnl  = 0.0
        to_close = []
        for ticker, pos in open_pos.items():
            df = data.get(ticker)
            if df is None or date not in df.index: continue
            bar   = df.loc[date]
            low   = float(bar["Low"])
            high  = float(bar["High"])
            close = float(bar["Close"])
            pos["days_held"] += 1
            reason = ep = None
            if   np.isfinite(low)  and low  <= pos["sl_price"]: reason, ep = "SL",      pos["sl_price"]
            elif np.isfinite(high) and high >= pos["tp_price"]: reason, ep = "TP",      pos["tp_price"]
            elif pos["days_held"] >= MAX_HOLD:                  reason, ep = "MaxHold", close
            if reason:
                pnl = (ep - pos["entry_price"]) * pos["shares"]
                day_pnl += pnl
                trades.append(dict(
                    ticker      = ticker,
                    entry_date  = pos["entry_date"].date(),
                    exit_date   = date.date(),
                    entry_price = round(pos["entry_price"], 2),
                    exit_price  = round(ep, 2),
                    pnl         = round(float(pnl), 2),
                    return_pct  = round((ep / pos["entry_price"] - 1) * 100, 2),
                    reason      = reason,
                    days_held   = int(pos["days_held"]),
                ))
                to_close.append(ticker)

        daily_pnl[di] = day_pnl
        for t in to_close:
            open_pos.pop(t, None)

        # ── Signal scan at close ──────────────────────────────────────────────
        if di < n_dates - 1:
            next_date = dates[di + 1]
            for ticker, df in data.items():
                if ticker in open_pos: continue
                if date not in df.index or dates[di-1] not in df.index: continue
                neck = find_neckline(df, df.index.get_loc(date))
                if neck is None: continue
                ct = float(df["Close"].iloc[df.index.get_loc(date)])
                cy = float(df["Close"].iloc[df.index.get_loc(date) - 1])
                if not (cy <= neck < ct): continue
                sma = sma_cache[ticker].iloc[df.index.get_loc(date)]
                if pd.isna(sma) or ct <= float(sma): continue
                strength = (ct - neck) / neck
                pending.setdefault(next_date, []).append((ticker, strength, neck))

    # ── Open positions snapshot ───────────────────────────────────────────────
    latest = dates[-1]
    pos_rows = []
    for ticker, pos in open_pos.items():
        df = data.get(ticker)
        if df is None or latest not in df.index: continue
        lc  = float(df.loc[latest, "Close"])
        ret = (lc / pos["entry_price"] - 1) * 100
        # Max-hold exit date estimate
        remaining     = MAX_HOLD - pos["days_held"]
        mh_exit_est   = (pd.Timestamp(latest) + BDay(remaining)).date()
        pos_rows.append(dict(
            ticker          = ticker,
            entry_date      = pos["entry_date"].date(),
            entry_price     = round(pos["entry_price"], 2),
            neckline        = round(pos["neckline"], 2),
            sl_price        = round(pos["sl_price"], 2),
            tp_price        = round(pos["tp_price"], 2),
            last_close      = round(lc, 2),
            days_held       = pos["days_held"],
            days_remaining  = remaining,
            mh_exit_est     = mh_exit_est,
            unreal_pct      = round(float(ret), 2),
            unreal_pnl      = round((lc - pos["entry_price"]) * pos["shares"], 2),
        ))
    positions_df = (pd.DataFrame(pos_rows).sort_values("unreal_pct", ascending=False)
                    if pos_rows else pd.DataFrame())

    # ── Action sheet: entries for next open ───────────────────────────────────
    # Re-scan latest close for fresh signals
    action_entries = []
    action_exits   = []
    if len(dates) >= 2:
        lat = dates[-1]
        prv = dates[-2]
        for ticker, df in data.items():
            if ticker in open_pos: continue
            if lat not in df.index or prv not in df.index: continue
            di   = df.index.get_loc(lat)
            neck = find_neckline(df, di)
            if neck is None: continue
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            if not (cy <= neck < ct): continue
            sma = sma_cache[ticker].iloc[di]
            if pd.isna(sma) or ct <= float(sma): continue
            action_entries.append(dict(
                ticker          = ticker,
                signal_date     = lat.date(),
                ref_close       = round(ct, 2),
                neckline        = round(neck, 2),
                est_sl          = round(ct * (1 - SL), 2),   # estimates based on close
                est_tp          = round(ct * (1 + TP), 2),   # recalculate from actual fill
                strength_pct    = round((ct - neck) / neck * 100, 2),
            ))
        # MaxHold exits
        for ticker, pos in open_pos.items():
            if pos["days_held"] >= MAX_HOLD:
                action_exits.append(dict(ticker=ticker, reason="MaxHold",
                                         entry_date=pos["entry_date"].date(),
                                         days_held=pos["days_held"]))

    enter_df = pd.DataFrame(action_entries)
    if not enter_df.empty:
        enter_df = enter_df.sort_values("strength_pct", ascending=False)
    exit_df = pd.DataFrame(action_exits)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["exit_date","ticker"], ascending=[False,True])

    return trades_df, positions_df, daily_pnl, latest.date(), enter_df, exit_df


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(trades_df, daily_pnl, n_dates):
    if trades_df.empty:
        return {}
    pnls    = trades_df["pnl"].values
    n       = len(trades_df)
    wins    = int((pnls > 0).sum())
    capital = NOTIONAL * MAX_POSITIONS
    n_years = n_dates / TRADING_DAYS
    total   = float(pnls.sum())
    cagr    = ((1 + total / capital) ** (1 / n_years) - 1) * 100
    cum     = np.cumsum(daily_pnl)
    std     = daily_pnl.std()
    sharpe  = daily_pnl.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else 0
    peak    = np.maximum.accumulate(cum)
    max_dd  = float((cum - peak).min())
    calmar  = cagr / abs(max_dd / capital * 100) if max_dd else 0
    reasons = trades_df["reason"].value_counts()
    avg_hold= trades_df["days_held"].mean()
    return dict(
        n        = n,
        wins     = wins,
        wr       = round(wins / n * 100, 1),
        avg_pnl  = round(total / n, 2),
        total    = round(total, 2),
        cagr     = round(cagr, 2),
        sharpe   = round(sharpe, 3),
        calmar   = round(calmar, 3),
        max_dd   = round(max_dd, 2),
        avg_hold = round(avg_hold, 1),
        pct_sl   = round(reasons.get("SL",      0) / n * 100, 1),
        pct_tp   = round(reasons.get("TP",      0) / n * 100, 1),
        pct_mh   = round(reasons.get("MaxHold", 0) / n * 100, 1),
    )


# ── Display ───────────────────────────────────────────────────────────────────

SEP  = "─" * 65
SEP2 = "═" * 65

def section(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def print_action_sheet(enter_df, exit_df, positions_df, asof):
    section(f"📋  TODAY'S ACTION SHEET  (as-of close: {asof})")
    slots = MAX_POSITIONS - (0 if positions_df.empty else len(positions_df))

    # Exits
    print(f"\n{'EXIT NEXT OPEN':}")
    if exit_df.empty:
        print("  None")
    else:
        for _, r in exit_df.iterrows():
            print(f"  SELL  {r.ticker:<6}  reason: {r.reason}  "
                  f"held {r.days_held} sessions")

    # Entries
    print(f"\nENTER NEXT OPEN  ({slots} slot{'s' if slots != 1 else ''} available)")
    if enter_df.empty or slots <= 0:
        if slots <= 0:
            print("  None — max positions reached")
        else:
            print("  None — no signals today")
    else:
        show = enter_df.head(slots)
        print(f"  ⚠  Recalculate SL/TP from actual fill price, not ref_close")
        print()
        hdr = f"  {'Ticker':<7} {'Neckline':>9} {'Ref Close':>10} {'Est SL':>9} {'Est TP':>9} {'Strength':>9}"
        print(hdr)
        print("  " + SEP)
        for _, r in show.iterrows():
            print(f"  {r.ticker:<7} {r.neckline:>9.2f} {r.ref_close:>10.2f} "
                  f"{r.est_sl:>9.2f} {r.est_tp:>9.2f} {r.strength_pct:>8.2f}%")
        if len(enter_df) > slots:
            print(f"\n  (+{len(enter_df)-slots} additional signals skipped — max positions)")


def print_open_positions(positions_df):
    section("📂  OPEN POSITIONS")
    if positions_df.empty:
        print("  None")
        return
    hdr = (f"  {'Ticker':<7} {'Entry':>10} {'Neckline':>9} {'SL':>8} "
           f"{'TP':>8} {'Last':>8} {'Held':>5} {'Left':>5} "
           f"{'Unreal%':>8} {'Unreal$':>9} {'MH Exit':>11}")
    print(hdr)
    print("  " + SEP)
    for _, r in positions_df.iterrows():
        flag = " ⚠" if r.days_remaining <= 3 else ""
        print(f"  {r.ticker:<7} {r.entry_price:>10.2f} {r.neckline:>9.2f} "
              f"{r.sl_price:>8.2f} {r.tp_price:>8.2f} {r.last_close:>8.2f} "
              f"{r.days_held:>5} {r.days_remaining:>5} "
              f"{r.unreal_pct:>7.1f}% {r.unreal_pnl:>9.2f} "
              f"{str(r.mh_exit_est):>11}{flag}")


def print_backtest_summary(m, n_dates):
    section("📊  BACKTEST SUMMARY  (5yr, open-confirmation entry)")
    if not m:
        print("  No trades"); return
    capital = NOTIONAL * MAX_POSITIONS
    print(f"""
  Period          : {n_dates} trading days (~{n_dates/TRADING_DAYS:.1f} years)
  Universe        : see tickers.txt
  Config          : TOL={TOL} | SMA{TREND_SMA} | SL={SL*100:.0f}% | TP={TP*100:.0f}% | MaxHold={MAX_HOLD}

  Total trades    : {m['n']}
  Win rate        : {m['wr']}%   (breakeven at {round(SL/(SL+TP)*100,1)}%)
  Avg P&L/trade   : ${m['avg_pnl']}
  Total P&L       : ${m['total']}
  CAGR            : {m['cagr']}%  (on ${capital:,.0f} max capital)
  Sharpe ratio    : {m['sharpe']}
  Calmar ratio    : {m['calmar']}
  Max drawdown    : ${m['max_dd']}
  Avg hold        : {m['avg_hold']} sessions

  Exit breakdown  : SL {m['pct_sl']}%  |  TP {m['pct_tp']}%  |  MaxHold {m['pct_mh']}%
""")


def print_recent_trades(trades_df, n=10):
    section(f"📒  LATEST {n} CLOSED TRADES")
    if trades_df.empty:
        print("  No closed trades"); return
    cols = ["ticker","entry_date","exit_date","entry_price",
            "exit_price","pnl","return_pct","reason","days_held"]
    print(trades_df.head(n)[cols].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0       = time.time()
    universe = load_universe()

    print(SEP2)
    print("  📉  DOUBLE BOTTOM SCANNER v5.2  —  Production")
    print(SEP2)
    print(f"  Universe : {len(universe)} tickers")
    print(f"  Config   : TOL={TOL} | SMA{TREND_SMA} | open-confirm | "
          f"SL={SL*100:.0f}% | TP={TP*100:.0f}% | MaxHold={MAX_HOLD}")
    print()

    data = download_data(universe)
    date_sets = [set(df.index) for df in data.values()]
    if not date_sets:
        print("ERROR: No usable data."); return
    dates = pd.DatetimeIndex(sorted(set.intersection(*date_sets)))
    print(f"\n  Usable: {len(data)} tickers | {len(dates)} trading days\n")

    print("Running backtest...")
    trades_df, positions_df, daily_pnl, asof, enter_df, exit_df = run_backtest(data, dates)
    m = compute_metrics(trades_df, daily_pnl, len(dates))
    print(f"  Done in {time.time()-t0:.1f}s\n")

    print_action_sheet(enter_df, exit_df, positions_df, asof)
    print_open_positions(positions_df)
    print_backtest_summary(m, len(dates))
    print_recent_trades(trades_df, 10)

    if not trades_df.empty:
        trades_df.to_csv("double_bottom_trades.csv", index=False)
        print(f"\n  Exported: double_bottom_trades.csv ({len(trades_df)} trades)")

    print(f"\n  Total runtime: {(time.time()-t0):.1f}s")
    print(SEP2)


if __name__ == "__main__":
    main()
