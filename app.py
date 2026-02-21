import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nselib import capital_market

st.set_page_config(page_title="NSE Equity Research", layout="wide")
st.title("📊 NSE Equity Research Dashboard")

# ---------------------------------------------------------
# LOAD NSE STOCK LIST
# ---------------------------------------------------------
@st.cache_data
def get_all_stocks():
    data = capital_market.equity_list()
    data = data[data[" SERIES"] == "EQ"]
    return sorted(data["SYMBOL"].unique())

stocks = get_all_stocks()
selected_stock = st.selectbox("Select NSE Stock", stocks)

# ---------------------------------------------------------
# SAFE INFO FETCH (CACHED)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        return yf.Ticker(symbol + ".NS").info
    except:
        return {}

# ---------------------------------------------------------
# OPTIMIZED PEER FINDER
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_sector_peers(selected_symbol, stock_list, max_peers=8):

    selected_info = get_stock_info(selected_symbol)
    sector = selected_info.get("sector")

    if not sector:
        return [], None

    peers = []

    for stock in stock_list:
        if stock == selected_symbol:
            continue

        info = get_stock_info(stock)
        if info.get("sector") == sector:
            peers.append(stock)

        if len(peers) >= max_peers:
            break

    return peers, sector


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if selected_stock:

    info = get_stock_info(selected_stock)
    ticker = yf.Ticker(selected_stock + ".NS")

    st.header(f"🔎 {selected_stock} Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Price", info.get("currentPrice", "N/A"))
    col2.metric("PE", info.get("trailingPE", "N/A"))
    col3.metric("Market Cap", info.get("marketCap", "N/A"))
    col4.metric("Sector", info.get("sector", "N/A"))

    # ---------------------------------------------------------
    # PRICE TREND (5Y)
    # ---------------------------------------------------------
    st.subheader("📈 Share Price (5Y)")

    price = ticker.history(period="5y")

    if not price.empty:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=price.index,
            y=price["Close"],
            name="Price"
        ))
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("No price data available.")

    # ---------------------------------------------------------
    # PE TREND
    # ---------------------------------------------------------
    st.subheader("📊 PE Trend")

    eps = info.get("trailingEps")

    if eps and eps > 0 and not price.empty:
        price["PE"] = price["Close"] / eps

        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=price.index,
            y=price["PE"],
            name="PE"
        ))
        st.plotly_chart(fig_pe, use_container_width=True)
    else:
        st.warning("PE trend unavailable.")

    # ---------------------------------------------------------
    # PEERS
    # ---------------------------------------------------------
    st.subheader("🏢 Same Sector Peers")

    peers, sector = get_sector_peers(selected_stock, stocks)

    if peers:

        tickers = [p + ".NS" for p in peers]

        # Batch latest price
        try:
            batch_price = yf.download(tickers, period="1d", progress=False)["Close"]
        except:
            batch_price = pd.DataFrame()

        peer_rows = []

        for peer in peers:
            info_peer = get_stock_info(peer)

            try:
                if len(tickers) == 1:
                    latest_price = batch_price.iloc[-1]
                else:
                    latest_price = batch_price[peer + ".NS"].iloc[-1]
            except:
                latest_price = None

            peer_rows.append({
                "Stock": peer,
                "Price": latest_price,
                "PE": info_peer.get("trailingPE"),
                "MarketCap": info_peer.get("marketCap")
            })

        peers_df = pd.DataFrame(peer_rows).dropna()

        if not peers_df.empty:
            st.dataframe(peers_df)

            # PE Comparison Chart
            fig_peer = go.Figure()
            fig_peer.add_trace(go.Bar(
                x=peers_df["Stock"],
                y=peers_df["PE"]
            ))
            st.plotly_chart(fig_peer, use_container_width=True)

        # ---------------------------------------------------------
        # SECTOR PERFORMANCE (Batch 1Y Return)
        # ---------------------------------------------------------
        st.subheader("📈 Sector vs Stock (1Y Return)")

        try:
            all_tickers = [selected_stock + ".NS"] + tickers
            batch_hist = yf.download(all_tickers, period="1y", progress=False)["Close"]

            returns = []

            # Selected stock return
            sel_series = batch_hist[selected_stock + ".NS"]
            sel_ret = (sel_series.iloc[-1] / sel_series.iloc[0] - 1) * 100
            returns.append({"Name": selected_stock, "Return": sel_ret})

            # Sector median return
            sector_returns = []

            for peer in peers:
                peer_series = batch_hist[peer + ".NS"]
                ret = (peer_series.iloc[-1] / peer_series.iloc[0] - 1) * 100
                sector_returns.append(ret)

            if sector_returns:
                median_ret = np.median(sector_returns)
                returns.append({"Name": f"{sector} Median", "Return": median_ret})

                perf_df = pd.DataFrame(returns)

                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(
                    x=perf_df["Name"],
                    y=perf_df["Return"]
                ))
                st.plotly_chart(fig_perf, use_container_width=True)

        except:
            st.warning("Sector performance unavailable.")

    else:
        st.warning("Peers not found.")

    # ---------------------------------------------------------
    # NEWS
    # ---------------------------------------------------------
    st.subheader("📰 Latest News")

    try:
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
    except:
        st.info("News unavailable.")