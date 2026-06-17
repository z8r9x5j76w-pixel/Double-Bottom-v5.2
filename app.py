"""
Double Bottom Scanner v5.2 — Streamlit App
==========================================
Two completely separate sections:

SECTION 1 — LIVE SIGNALS (today's actionable trades)
  Scans the latest close for fresh breakouts.
  These are NEW signals to consider for next open entry.
  No position limit applied — shows ALL signals that fired.

SECTION 2 — BACKTEST TRACKER (historical simulation)
  Shows what the backtest engine currently holds.
  For reference/validation only — NOT your live portfolio.
  Helps you understand strategy health and current drawdown.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from double_bottom_scanner_v5_2 import (
    load_universe, download_data, run_backtest, compute_metrics,
    find_neckline, SL, TP, MAX_HOLD, TREND_SMA, TOL, NOTIONAL, MAX_POSITIONS, TRADING_DAYS
)

st.set_page_config(
    page_title="Double Bottom Scanner v5.2",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .live-signal-box {
        background: #052e16;
        border: 1.5px solid #16a34a;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .backtest-box {
        background: #1e1b4b;
        border: 1px solid #4f46e5;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .warn-box {
        background: #431407;
        border: 1px solid #f97316;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .section-label-live {
        background: #16a34a;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .section-label-bt {
        background: #4f46e5;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Double Bottom v5.2")
    st.markdown(f"""
    **Config (locked)**
    - TOL: `{TOL}` | Trend: SMA{TREND_SMA}
    - Entry: open > neckline ✅
    - SL: `{SL*100:.0f}%` | TP: `{TP*100:.0f}%`
    - MaxHold: `{MAX_HOLD}` sessions
    - Notional: `${NOTIONAL:,.0f}` / trade
    """)
    st.divider()
    st.markdown("""
    **Backtest (5yr)**
    Sharpe `1.858` | CAGR `11.17%`
    Calmar `1.27` | MaxDD `$-1,760`
    Win rate `29.3%` *(BE: 22.2%)*
    """)
    st.divider()
    run_btn = st.button("🔄  Run Scanner", type="primary", use_container_width=True)
    st.caption("Downloads 5yr daily data + runs backtest. ~2–3 min first run, cached 1hr after.")


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_full_scan():
    universe  = load_universe()
    data      = download_data(universe)
    date_sets = [set(df.index) for df in data.values()]
    dates     = pd.DatetimeIndex(sorted(set.intersection(*date_sets)))

    trades_df, positions_df, daily_pnl, asof, enter_df, exit_df = \
        run_backtest(data, dates)
    m = compute_metrics(trades_df, daily_pnl, len(dates))

    # ── LIVE SIGNALS: fresh scan of latest close, NO position cap ────────────
    sma_cache = {t: df["Close"].rolling(TREND_SMA).mean() for t, df in data.items()}
    live_signals = []
    if len(dates) >= 2:
        lat = dates[-1]
        for ticker, df in data.items():
            if lat not in df.index: continue
            di = df.index.get_loc(lat)
            if di < 1: continue
            neck = find_neckline(df, di)
            if neck is None: continue
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            if not (cy <= neck < ct): continue
            sma = sma_cache[ticker].iloc[di]
            if pd.isna(sma) or ct <= float(sma): continue
            strength = (ct - neck) / neck
            live_signals.append(dict(
                ticker       = ticker,
                signal_date  = lat.date(),
                neckline     = round(neck, 2),
                ref_close    = round(ct, 2),
                est_sl       = round(ct * (1 - SL), 2),
                est_tp       = round(ct * (1 + TP), 2),
                strength_pct = round(strength * 100, 2),
                note         = "⚠ Recalculate SL/TP from actual fill at open",
            ))
    live_df = pd.DataFrame(live_signals).sort_values(
        "strength_pct", ascending=False) if live_signals else pd.DataFrame()

    equity = pd.DataFrame({
        "Date": dates,
        "Cumulative P&L ($)": np.cumsum(daily_pnl),
    }).set_index("Date")

    return data, dates, trades_df, positions_df, daily_pnl, asof, \
           enter_df, exit_df, m, equity, live_df


# ── Gate ──────────────────────────────────────────────────────────────────────
st.title("📉 Double Bottom Scanner v5.2")

if not run_btn and "scan_done" not in st.session_state:
    st.info("Click **Run Scanner** in the sidebar to load data.")
    st.stop()

if run_btn or "scan_done" not in st.session_state:
    with st.spinner("Downloading data and running backtest… (~2 min)"):
        try:
            result = run_full_scan()
            st.session_state["scan_done"]   = True
            st.session_state["scan_result"] = result
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

(data, dates, trades_df, positions_df, daily_pnl,
 asof, enter_df, exit_df, m, equity, live_df) = st.session_state["scan_result"]

st.caption(f"Data as-of close: **{asof}**  |  Today's signals based on this close.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LIVE SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<span class="section-label-live">🟢 LIVE SIGNALS</span>', unsafe_allow_html=True)
st.subheader("Today's Actionable Signals — Enter at Next Open")
st.markdown(
    "These are **fresh breakouts detected on today's close**. "
    "Each ticker has crossed its double-bottom neckline with SMA50 confirmation. "
    "**No position limit applied** — all valid signals shown. You decide how many to take."
)

if live_df.empty:
    if asof == pd.Timestamp.now().date():
        st.info("No double-bottom breakouts detected on today's close.")
    else:
        st.warning(
            f"Signal date is **{asof}** — market may not have closed yet today. "
            "Re-run after 4pm ET to get today's signals."
        )
else:
    st.markdown(
        '<div class="warn-box">⚠️ <strong>Est. SL/TP use today\'s close as proxy. '
        'Always recalculate from your actual fill price at open before placing bracket orders.'
        '</strong></div>',
        unsafe_allow_html=True
    )
    st.dataframe(
        live_df[["ticker","signal_date","neckline","ref_close","est_sl","est_tp","strength_pct"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker":       st.column_config.TextColumn("Ticker"),
            "signal_date":  st.column_config.TextColumn("Signal Date"),
            "neckline":     st.column_config.NumberColumn("Neckline",    format="$%.2f"),
            "ref_close":    st.column_config.NumberColumn("Ref Close",   format="$%.2f"),
            "est_sl":       st.column_config.NumberColumn("Est SL (2%)", format="$%.2f"),
            "est_tp":       st.column_config.NumberColumn("Est TP (7%)", format="$%.2f"),
            "strength_pct": st.column_config.NumberColumn("Strength",    format="%.2f%%"),
        }
    )
    st.caption(
        f"**How to use:** At market open tomorrow, check if open price > neckline. "
        f"If yes → enter. If open gaps back below neckline → skip (open-confirmation rule). "
        f"Set SL = fill × 0.98, TP = fill × 1.07, exit by session {MAX_HOLD} if neither hit."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BACKTEST TRACKER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<span class="section-label-bt">🔵 BACKTEST TRACKER</span>', unsafe_allow_html=True)
st.subheader("Backtest Simulation — Historical Reference Only")
st.markdown(
    "**This is not your live portfolio.** "
    "These are positions held by the backtest simulation as of the latest date. "
    "They show how the strategy *would* be positioned if run mechanically since inception. "
    "Use this to validate strategy health and track paper performance."
)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total P&L",    f"${m['total']:,.0f}")
k2.metric("CAGR",         f"{m['cagr']}%")
k3.metric("Sharpe",       f"{m['sharpe']}")
k4.metric("Calmar",       f"{m['calmar']}")
k5.metric("Max Drawdown", f"${m['max_dd']:,.0f}")
k6.metric("Win Rate",     f"{m['wr']}%")

# Backtest open positions
st.markdown("#### Backtest Open Positions")
st.caption("Positions currently open in the simulation. Progress bar = sessions used / 30 max.")

if positions_df.empty:
    st.info("No open positions in backtest simulation.")
else:
    # Add progress column
    pos_display = positions_df.copy()
    pos_display["progress"] = pos_display["days_held"] / MAX_HOLD

    def colour_pnl(val):
        if isinstance(val, (int, float)):
            color = "#16a34a" if val > 0 else "#dc2626" if val < 0 else "#94a3b8"
            return f"color: {color}; font-weight: 600"
        return ""

    st.dataframe(
        pos_display[[
            "ticker","entry_date","entry_price","neckline",
            "sl_price","tp_price","last_close",
            "days_held","days_remaining","mh_exit_est",
            "unreal_pct","unreal_pnl","progress"
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker":         st.column_config.TextColumn("Ticker"),
            "entry_date":     st.column_config.TextColumn("Entry Date"),
            "entry_price":    st.column_config.NumberColumn("Entry",     format="$%.2f"),
            "neckline":       st.column_config.NumberColumn("Neckline",  format="$%.2f"),
            "sl_price":       st.column_config.NumberColumn("SL",        format="$%.2f"),
            "tp_price":       st.column_config.NumberColumn("TP",        format="$%.2f"),
            "last_close":     st.column_config.NumberColumn("Last",      format="$%.2f"),
            "days_held":      st.column_config.NumberColumn("Held",      format="%d sessions"),
            "days_remaining": st.column_config.NumberColumn("Remaining", format="%d sessions"),
            "mh_exit_est":    st.column_config.TextColumn("MH Exit"),
            "unreal_pct":     st.column_config.NumberColumn("Unreal %",  format="%.1f%%"),
            "unreal_pnl":     st.column_config.NumberColumn("Unreal $",  format="$%.2f"),
            "progress":       st.column_config.ProgressColumn(
                                "Hold Progress", min_value=0, max_value=1, format="%.0%%"),
        }
    )

# Equity curve
st.markdown("#### Backtest Equity Curve")
st.line_chart(equity, use_container_width=True, color="#4f46e5")

# Trade log
st.markdown("#### Backtest Trade Log")
tab_recent, tab_winners, tab_losers, tab_full = st.tabs(
    ["Latest 20", "Top Winners", "Worst Losers", "Full Log"])

with tab_recent:
    st.dataframe(trades_df.head(20), use_container_width=True, hide_index=True)
with tab_winners:
    st.dataframe(
        trades_df[trades_df["pnl"] > 0].sort_values("pnl", ascending=False).head(20),
        use_container_width=True, hide_index=True)
with tab_losers:
    st.dataframe(
        trades_df[trades_df["pnl"] < 0].sort_values("pnl").head(20),
        use_container_width=True, hide_index=True)
with tab_full:
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV", trades_df.to_csv(index=False).encode(),
        "double_bottom_trades.csv", "text/csv")

# Monthly P&L
st.markdown("#### Monthly P&L (Backtest)")
monthly = (trades_df
    .assign(month=pd.to_datetime(trades_df["exit_date"]).dt.to_period("M"))
    .groupby("month")["pnl"].sum()
    .reset_index()
    .assign(month=lambda x: x["month"].astype(str),
            colour=lambda x: x["pnl"].apply(lambda v: "🟢" if v >= 0 else "🔴")))
monthly.columns = ["Month","P&L ($)",""]
st.dataframe(
    monthly.sort_values("Month", ascending=False).head(24),
    use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "📉 Double Bottom Scanner v5.2 | Research only — not financial advice. "
    f"Config: TOL={TOL} | SMA{TREND_SMA} | open-confirm | "
    f"SL={SL*100:.0f}% | TP={TP*100:.0f}% | MaxHold={MAX_HOLD}"
)
