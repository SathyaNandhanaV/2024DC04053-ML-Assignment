import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nselib import capital_market

st.set_page_config(page_title="NSE Equity Research Stable", layout="wide")
st.title("📊 NSE Equity Research Dashboard (Stable Version)")

# ---------------------------------------------------------
# LOAD STOCK LIST
# ---------------------------------------------------------
@st.cache_data
def get_all_stocks():
    data = capital_market.equity_list()
    data = data[data[" SERIES"] == "EQ"]
    return sorted(data["SYMBOL"].unique())

stocks = get_all_stocks()

selected_stock = st.selectbox("Select NSE Stock", stocks)

# ---------------------------------------------------------
# GET BASIC STOCK INFO (CACHED)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_basic_info(symbol):
    t = yf.Ticker(symbol + ".NS")
    return t.info

# ---------------------------------------------------------
# GET PEERS (LIMITED + CACHED)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_sector_peers(selected, stock_list, max_peers=10):

    selected_info = get_basic_info(selected)
    sector = selected_info.get("sector")

    if not sector:
        return [], None

    peers = []

    for stock in stock_list:
        if stock == selected:
            continue

        try:
            info = get_basic_info(stock)
            if info.get("sector") == sector:
                peers.append(stock)

            if len(peers) >= max_peers:
                break
        except:
            continue

    return peers, sector


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if selected_stock:

    info = get_basic_info(selected_stock)
    ticker = yf.Ticker(selected_stock + ".NS")

    st.header(f"🔎 {selected_stock} Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Price", info.get("currentPrice", "N/A"))
    col2.metric("PE", info.get("trailingPE", "N/A"))
    col3.metric("Market Cap", info.get("marketCap", "N/A"))
    col4.metric("Sector", info.get("sector", "N/A"))

    # ---------------------------------------------------------
    # PRICE TREND
    # ---------------------------------------------------------
    st.subheader("📈 Share Price (5Y)")

    price = ticker.history(period="5y")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price.index,
        y=price["Close"],
        name="Price"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # PE TREND
    # ---------------------------------------------------------
    st.subheader("📊 PE Trend")

    eps = info.get("trailingEps")

    if eps and eps > 0:
        price["PE"] = price["Close"] / eps

        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=price.index,
            y=price["PE"],
            name="PE"
        ))

        st.plotly_chart(fig_pe, use_container_width=True)
    else:
        st.warning("EPS unavailable.")

    # ---------------------------------------------------------
    # PEERS
    # ---------------------------------------------------------
    st.subheader("🏢 Same Sector Peers")

    peers, sector = get_sector_peers(selected_stock, stocks)

    if peers:

        # Batch price download
        tickers = [p + ".NS" for p in peers]
        batch_prices = yf.download(tickers, period="1d")["Close"]

        peer_data = []

        for peer in peers:
            info_peer = get_basic_info(peer)

            try:
                latest_price = batch_prices[peer + ".NS"].iloc[-1]
            except:
                latest_price = None

            peer_data.append({
                "Stock": peer,
                "Price": latest_price,
                "PE": info_peer.get("trailingPE"),
                "MarketCap": info_peer.get("marketCap")
            })

        peers_df = pd.DataFrame(peer_data).dropna()

        st.dataframe(peers_df)

        # PE Chart
        fig_peer = go.Figure()
        fig_peer.add_trace(go.Bar(
            x=peers_df["Stock"],
            y=peers_df["PE"]
        ))

        st.plotly_chart(fig_peer, use_container_width=True)

        # ---------------------------------------------------------
        # SECTOR PERFORMANCE
        # ---------------------------------------------------------
        st.subheader("📈 Sector vs Stock (1Y Return)")

        returns = []

        sel_hist = ticker.history(period="1y")
        if len(sel_hist) > 0:
            sel_ret = (sel_hist["Close"].iloc[-1] /
                       sel_hist["Close"].iloc[0] - 1) * 100
            returns.append({"Name": selected_stock, "Return": sel_ret})

        sector_returns = []

        for peer in peers:
            t = yf.Ticker(peer + ".NS")
            hist = t.history(period="1y")
            if len(hist) > 0:
                ret = (hist["Close"].iloc[-1] /
                       hist["Close"].iloc[0] - 1) * 100
                sector_returns.append(ret)

        if sector_returns:
            median_ret = np.median(sector_returns)
            returns.append({"Name": f"{sector} Sector Median", "Return": median_ret})

            perf_df = pd.DataFrame(returns)

            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                x=perf_df["Name"],
                y=perf_df["Return"]
            ))

            st.plotly_chart(fig_perf, use_container_width=True)

    else:
        st.warning("Peers not found.")

    # ---------------------------------------------------------
    # NEWS
    # ---------------------------------------------------------
    st.subheader("📰 Latest News")

    news = ticker.news

    if news:
        for item in news[:5]:
            title = item.get("title")
            link = item.get("link")
            if title and link:
                st.markdown(f"### [{title}]({link})")
                st.write("---")
    else:
        st.info("No recent news.")