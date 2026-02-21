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

# ---------------------------------------------------------
# 🔎 SEARCH BOX
# ---------------------------------------------------------
search_text = st.text_input("🔎 Search Stock (Type Name or Symbol)")

if search_text:
    filtered_stocks = [s for s in stocks if search_text.upper() in s.upper()]
else:
    filtered_stocks = stocks

selected_stock = st.selectbox("Select NSE Stock", filtered_stocks)

# ---------------------------------------------------------
# SAFE INFO FETCH
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        return yf.Ticker(symbol + ".NS").info
    except:
        return {}

# ---------------------------------------------------------
# PEER FINDER
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
    # PRICE TREND
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

            fig_peer = go.Figure()
            fig_peer.add_trace(go.Bar(
                x=peers_df["Stock"],
                y=peers_df["PE"]
            ))
            st.plotly_chart(fig_peer, use_container_width=True)

    else:
        st.warning("Peers not found.")

    # ---------------------------------------------------------
    # 📰 FILTERED NEWS SECTION
    # ---------------------------------------------------------
    st.subheader("📰 Latest News (Filtered by Stock Name)")

    try:
        news = ticker.news
        found = False

        if news:
            for item in news:
                title = item.get("title")
                link = item.get("link")

                # Only show articles containing stock name
                if title and selected_stock.lower() in title.lower():
                    st.markdown(f"### [{title}]({link})")
                    st.write("---")
                    found = True

        if not found:
            st.info("No recent articles mentioning this stock directly.")

    except:
        st.info("News unavailable.")