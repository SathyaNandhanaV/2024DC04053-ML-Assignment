import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nselib import capital_market

st.set_page_config(page_title="NSE Equity Research Pro", layout="wide")
st.title("📊 NSE Equity Research Dashboard (Ultra Optimized)")

# ----------------------------------------------------------
# LOAD NSE STOCK LIST
# ----------------------------------------------------------
@st.cache_data
def get_all_nse_stocks():
    data = capital_market.equity_list()
    data = data[data[" SERIES"] == "EQ"]
    return sorted(data["SYMBOL"].unique())

stocks = get_all_nse_stocks()

selected_stock = st.selectbox("Select NSE Stock", stocks)

# ----------------------------------------------------------
# OPTIMIZED PEER FINDER
# ----------------------------------------------------------
@st.cache_data
def get_sector_peers(selected_stock, stock_list, max_peers=20):

    selected_ticker = yf.Ticker(selected_stock + ".NS")
    selected_sector = selected_ticker.info.get("sector")

    if not selected_sector:
        return [], None

    peers = []

    for stock in stock_list:
        if stock == selected_stock:
            continue

        try:
            t = yf.Ticker(stock + ".NS")
            sector = t.info.get("sector")

            if sector == selected_sector:
                peers.append(stock)

            if len(peers) >= max_peers:
                break

        except:
            continue

    return peers, selected_sector


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if selected_stock:

    ticker = yf.Ticker(selected_stock + ".NS")
    info = ticker.info

    st.header(f"🔎 {selected_stock} Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", info.get("currentPrice", "N/A"))
    col2.metric("PE Ratio", info.get("trailingPE", "N/A"))
    col3.metric("Market Cap", info.get("marketCap", "N/A"))
    col4.metric("Sector", info.get("sector", "N/A"))

    # ----------------------------------------------------------
    # PRICE TREND
    # ----------------------------------------------------------
    st.subheader("📈 Share Price Trend (5Y)")
    price_data = ticker.history(period="5y")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data["Close"],
        name="Close Price"
    ))
    st.plotly_chart(fig_price, use_container_width=True)

    # ----------------------------------------------------------
    # PE TREND
    # ----------------------------------------------------------
    st.subheader("📊 PE Ratio Trend")

    eps = info.get("trailingEps")

    if eps and eps > 0:
        price_data["PE"] = price_data["Close"] / eps

        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data["PE"],
            name="PE Ratio"
        ))
        st.plotly_chart(fig_pe, use_container_width=True)
    else:
        st.warning("EPS not available.")

    # ----------------------------------------------------------
    # PEER ANALYSIS
    # ----------------------------------------------------------
    st.subheader("🏢 Peer Comparison (Same Sector)")

    peer_symbols, sector = get_sector_peers(selected_stock, stocks)

    if sector and peer_symbols:

        peer_tickers = [p + ".NS" for p in peer_symbols]

        # Batch download prices
        price_batch = yf.download(peer_tickers, period="1d")["Close"]

        peer_data = []

        for peer in peer_symbols:
            try:
                t = yf.Ticker(peer + ".NS")
                i = t.info

                peer_data.append({
                    "Stock": peer,
                    "Price": price_batch.get(peer + ".NS"),
                    "PE": i.get("trailingPE"),
                    "MarketCap": i.get("marketCap")
                })
            except:
                continue

        peers_df = pd.DataFrame(peer_data).dropna()

        if not peers_df.empty:
            peers_df = peers_df.sort_values("MarketCap", ascending=False)
            st.dataframe(peers_df)

            # PE comparison chart
            st.subheader("📊 Peer PE Comparison")

            fig_peer_pe = go.Figure()
            fig_peer_pe.add_trace(go.Bar(
                x=peers_df["Stock"],
                y=peers_df["PE"]
            ))
            st.plotly_chart(fig_peer_pe, use_container_width=True)

        # ----------------------------------------------------------
        # SECTOR PERFORMANCE
        # ----------------------------------------------------------
        st.subheader("📈 Sector Performance vs Selected Stock (1Y Return)")

        returns = []

        sel_hist = ticker.history(period="1y")
        if len(sel_hist) > 0:
            sel_return = (sel_hist["Close"].iloc[-1] /
                          sel_hist["Close"].iloc[0] - 1) * 100

            returns.append({
                "Name": selected_stock,
                "Return": sel_return
            })

        sector_returns = []

        for peer in peer_symbols:
            try:
                t = yf.Ticker(peer + ".NS")
                hist = t.history(period="1y")
                if len(hist) > 0:
                    ret = (hist["Close"].iloc[-1] /
                           hist["Close"].iloc[0] - 1) * 100
                    sector_returns.append(ret)
            except:
                continue

        if sector_returns:
            median_sector_return = np.median(sector_returns)

            returns.append({
                "Name": f"{sector} Sector (Median)",
                "Return": median_sector_return
            })

            perf_df = pd.DataFrame(returns)

            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                x=perf_df["Name"],
                y=perf_df["Return"]
            ))

            st.plotly_chart(fig_perf, use_container_width=True)

    else:
        st.warning("Sector data unavailable.")

    # ----------------------------------------------------------
    # NEWS (SAFE)
    # ----------------------------------------------------------
    st.subheader("📰 Latest News")

    news_items = ticker.news

    if news_items:
        for item in news_items[:5]:
            title = item.get("title")
            link = item.get("link")
            publisher = item.get("publisher")

            if title and link:
                st.markdown(f"### [{title}]({link})")
                if publisher:
                    st.caption(publisher)
                st.write("---")
    else:
        st.info("No recent news available.")