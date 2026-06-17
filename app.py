"""
Double Bottom Scanner v5.2 — Streamlit App
==========================================
Run locally:   streamlit run app.py
Deploy:        push to GitHub + connect to Streamlit Cloud

Expects tickers.txt in the same directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, time
from pathlib import Path

# ── Import scanner core ───────────────────────────────────────────────────────
# We import functions directly from the scanner module
sys.path.insert(0, str(Path(__file__).parent))
from double_bottom_scanner_v5_2 import (
    load_universe, download_data, run_backtest, compute_metrics,
    SL, TP, MAX_HOLD, TREND_SMA, TOL, NOTIONAL, MAX_POSITIONS, TRADING_DAYS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Double Bottom Scanner v5.2",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 3px solid #7c3aed;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .signal-box {
        background: #064e3b;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .warn-box {
        background: #431407;
        border: 1px solid #f97316;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Double Bottom v5.2")
    st.markdown(f"""
    **Config (locked)**
    - TOL: `{TOL}` (tight symmetry)
    - Trend: SMA{TREND_SMA}
    - Entry: open > neckline ✅
    - SL: `{SL*100:.0f}%` | TP: `{TP*100:.0f}%`
    - MaxHold: `{MAX_HOLD}` sessions
    - Notional: `${NOTIONAL:,.0f}` / trade
    - Max positions: `{MAX_POSITIONS}`
    """)
    st.divider()
    st.markdown("""
    **Backtest stats (5yr)**
    - Sharpe: `1.858`
    - CAGR: `11.17%`
    - Calmar: `1.27`
    - MaxDD: `$-1,760`
    - Win rate: `29.3%` *(BE: 22.2%)*
    """)
    st.divider()
    run_btn = st.button("🔄  Run Scanner", type="primary", use_container_width=True)
    st.caption("Downloads 5yr daily data + runs backtest. ~2–3 min.")


# ── Caching ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_full_scan():
    universe = load_universe()
    data     = download_data(universe)
    date_sets= [set(df.index) for df in data.values()]
    dates    = pd.DatetimeIndex(sorted(set.intersection(*date_sets)))
    trades_df, positions_df, daily_pnl, asof, enter_df, exit_df = run_backtest(data, dates)
    m        = compute_metrics(trades_df, daily_pnl, len(dates))
    equity   = pd.DataFrame({
        "Date": dates,
        "Cumulative P&L ($)": np.cumsum(daily_pnl)
    }).set_index("Date")
    return trades_df, positions_df, daily_pnl, asof, enter_df, exit_df, m, equity, len(dates)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📉 Double Bottom Scanner v5.2")
st.caption("Production | Config C + Open Confirmation | TOL=0.02 | SMA50 | SL=2% TP=7% MH=30")

if not run_btn and "scan_done" not in st.session_state:
    st.info("Click **Run Scanner** in the sidebar to fetch data and run the backtest.")
    st.stop()

# Run or use cache
if run_btn or "scan_done" not in st.session_state:
    with st.spinner("Downloading data and running backtest…"):
        try:
            result = run_full_scan()
            st.session_state["scan_done"]   = True
            st.session_state["scan_result"] = result
        except Exception as e:
            st.error(f"Scanner error: {e}")
            st.stop()

trades_df, positions_df, daily_pnl, asof, enter_df, exit_df, m, equity, n_dates = \
    st.session_state["scan_result"]

st.caption(f"Last run: as-of close **{asof}**")

# ── KPI row ───────────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total P&L",    f"${m['total']:,.0f}")
k2.metric("CAGR",         f"{m['cagr']}%")
k3.metric("Sharpe",       f"{m['sharpe']}")
k4.metric("Calmar",       f"{m['calmar']}")
k5.metric("Max Drawdown", f"${m['max_dd']:,.0f}")
k6.metric("Win Rate",     f"{m['wr']}%")

# ── Action sheet ──────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"📋 Action Sheet — {asof}")

col_exits, col_entries = st.columns([1, 2])

with col_exits:
    st.markdown("**Exit next open**")
    if exit_df.empty:
        st.success("No exits scheduled")
    else:
        for _, r in exit_df.iterrows():
            st.warning(f"SELL **{r.ticker}** — {r.reason} (held {r.days_held} sessions)")

with col_entries:
    slots = MAX_POSITIONS - (0 if positions_df.empty else len(positions_df))
    st.markdown(f"**Enter next open** — {slots} slot{'s' if slots != 1 else ''} available")
    if enter_df.empty or slots <= 0:
        if slots <= 0:
            st.info("Max positions reached — no new entries")
        else:
            st.info("No signals today")
    else:
        show = enter_df.head(slots).copy()
        st.markdown(
            '<div class="warn-box">⚠️ Est. SL/TP based on ref_close. '
            'Recalculate from actual fill price before placing orders.</div>',
            unsafe_allow_html=True)
        st.dataframe(
            show[["ticker","neckline","ref_close","est_sl","est_tp","strength_pct"]],
            use_container_width=True, hide_index=True,
            column_config={
                "ticker":       st.column_config.TextColumn("Ticker"),
                "neckline":     st.column_config.NumberColumn("Neckline",   format="$%.2f"),
                "ref_close":    st.column_config.NumberColumn("Ref Close",  format="$%.2f"),
                "est_sl":       st.column_config.NumberColumn("Est SL",     format="$%.2f"),
                "est_tp":       st.column_config.NumberColumn("Est TP",     format="$%.2f"),
                "strength_pct": st.column_config.NumberColumn("Strength %", format="%.2f%%"),
            }
        )

# ── Open positions ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📂 Open Positions")
if positions_df.empty:
    st.info("No open positions")
else:
    def colour_pnl(val):
        color = "#10b981" if val > 0 else "#ef4444" if val < 0 else "#94a3b8"
        return f"color: {color}; font-weight: 600"

    styled = positions_df.style.applymap(
        colour_pnl, subset=["unreal_pct","unreal_pnl"])
    st.dataframe(
        styled,
        use_container_width=True, hide_index=True,
        column_config={
            "ticker":         st.column_config.TextColumn("Ticker"),
            "entry_date":     st.column_config.TextColumn("Entry Date"),
            "entry_price":    st.column_config.NumberColumn("Entry",     format="$%.2f"),
            "neckline":       st.column_config.NumberColumn("Neckline",  format="$%.2f"),
            "sl_price":       st.column_config.NumberColumn("SL",        format="$%.2f"),
            "tp_price":       st.column_config.NumberColumn("TP",        format="$%.2f"),
            "last_close":     st.column_config.NumberColumn("Last",      format="$%.2f"),
            "days_held":      st.column_config.NumberColumn("Held",      format="%d"),
            "days_remaining": st.column_config.NumberColumn("Remaining", format="%d"),
            "mh_exit_est":    st.column_config.TextColumn("MH Exit"),
            "unreal_pct":     st.column_config.NumberColumn("Unreal %",  format="%.1f%%"),
            "unreal_pnl":     st.column_config.NumberColumn("Unreal $",  format="$%.2f"),
        }
    )

# ── Equity curve ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Equity Curve (Cumulative P&L)")
st.line_chart(equity, use_container_width=True, color="#7c3aed")

# ── Trade log ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📒 Trade Log")

tab_recent, tab_winners, tab_losers, tab_full = st.tabs(
    ["Latest 20", "Top Winners", "Worst Losers", "Full Log"])

with tab_recent:
    st.dataframe(trades_df.head(20), use_container_width=True, hide_index=True)

with tab_winners:
    top_w = trades_df[trades_df["pnl"] > 0].sort_values("pnl", ascending=False).head(20)
    st.dataframe(top_w, use_container_width=True, hide_index=True)

with tab_losers:
    top_l = trades_df[trades_df["pnl"] < 0].sort_values("pnl").head(20)
    st.dataframe(top_l, use_container_width=True, hide_index=True)

with tab_full:
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    csv = trades_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "double_bottom_trades.csv", "text/csv")

# ── Exit breakdown ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Exit Breakdown")
c1, c2 = st.columns(2)

with c1:
    reason_counts = trades_df["reason"].value_counts().reset_index()
    reason_counts.columns = ["Reason", "Count"]
    st.dataframe(reason_counts, use_container_width=True, hide_index=True)
    st.caption(f"SL {m['pct_sl']}%  |  TP {m['pct_tp']}%  |  MaxHold {m['pct_mh']}%")

with c2:
    monthly = (trades_df
               .assign(month=pd.to_datetime(trades_df["exit_date"]).dt.to_period("M"))
               .groupby("month")["pnl"].sum()
               .reset_index()
               .assign(month=lambda x: x["month"].astype(str)))
    monthly.columns = ["Month", "P&L ($)"]
    st.dataframe(monthly.sort_values("Month", ascending=False).head(24),
                 use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Double Bottom Scanner v5.2 | Research only — not financial advice. "
    f"Config: TOL={TOL} | SMA{TREND_SMA} | SL={SL*100:.0f}% | TP={TP*100:.0f}% | MaxHold={MAX_HOLD}"
)
