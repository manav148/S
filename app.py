#!/usr/bin/env python3
"""Streamlit UI for Stock/Crypto Recommender."""

import asyncio
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import config
from src.data_sources.betting import BettingMarketFetcher
from src.recommender import Recommendation, RecommendationEngine, RecommendationType

# Page config
st.set_page_config(
    page_title="Stock/Crypto Recommender",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Helper to run async functions
def run_async(coro):
    """Run an async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Cached data fetching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_top_picks(asset_type: str, top_n: int) -> list[dict]:
    """Fetch top picks and convert to serializable format."""

    async def fetch():
        engine = RecommendationEngine()
        try:
            symbols = await engine.discover_symbols(asset_type, limit=top_n * 2)
            recs = await engine.get_top_picks(symbols, asset_type, top_n=top_n)
            return [rec_to_dict(r) for r in recs]
        finally:
            await engine.close()

    return run_async(fetch())


@st.cache_data(ttl=300)
def analyze_symbol(symbol: str, asset_type: str) -> dict:
    """Analyze a single symbol."""

    async def fetch():
        engine = RecommendationEngine()
        try:
            rec = await engine.get_recommendation(symbol.upper(), asset_type)
            return rec_to_dict(rec)
        finally:
            await engine.close()

    return run_async(fetch())


@st.cache_data(ttl=300)
def get_polymarket_data(market_type: str, limit: int) -> list[dict]:
    """Fetch Polymarket data."""

    async def fetch():
        fetcher = BettingMarketFetcher()
        try:
            if market_type == "crypto":
                markets = await fetcher.get_crypto_markets(limit)
            elif market_type == "stock":
                markets = await fetcher.get_stock_markets(limit)
            else:
                markets = await fetcher.get_top_markets(limit)
            return [
                {
                    "title": m.title,
                    "probability": m.probability,
                    "liquidity": m.liquidity or 0,
                    "end_date": m.end_date.isoformat() if m.end_date else None,
                    "url": m.url,
                }
                for m in markets
            ]
        finally:
            await fetcher.close()

    return run_async(fetch())


@st.cache_data(ttl=300)
def search_markets(query: str, limit: int) -> list[dict]:
    """Search Polymarket."""

    async def fetch():
        fetcher = BettingMarketFetcher()
        try:
            markets = await fetcher.search_markets(query, limit)
            return [
                {
                    "title": m.title,
                    "probability": m.probability,
                    "liquidity": m.liquidity or 0,
                    "end_date": m.end_date.isoformat() if m.end_date else None,
                    "url": m.url,
                }
                for m in markets
            ]
        finally:
            await fetcher.close()

    return run_async(fetch())


def rec_to_dict(rec: Recommendation) -> dict:
    """Convert Recommendation to serializable dict."""
    result = {
        "symbol": rec.symbol,
        "asset_type": rec.asset_type,
        "recommendation": rec.recommendation.value,
        "overall_score": rec.overall_score,
        "confidence": rec.confidence,
        "analyst_score": rec.analyst_score.score if rec.analyst_score else 0.5,
        "analyst_confidence": rec.analyst_score.confidence if rec.analyst_score else "none",
        "analyst_count": rec.analyst_data.total_analysts if rec.analyst_data else 0,
        "betting_score": rec.betting_score.score if rec.betting_score else 0.5,
        "betting_confidence": rec.betting_score.confidence if rec.betting_score else "none",
        "news_score": rec.news_score.score if rec.news_score else 0.5,
        "news_confidence": rec.news_score.confidence if rec.news_score else "none",
        "options_score": rec.options_score.score if rec.options_score else 0.5,
        "options_confidence": rec.options_score.confidence if rec.options_score else "none",
        "supporting_factors": rec.supporting_factors,
        "risk_factors": rec.risk_factors,
        "current_price": rec.current_price,
        "target_price": rec.target_price,
        "upside_potential": rec.upside_potential,
        "downside_risk": rec.downside_risk,
        "risk_reward_ratio": rec.risk_reward_ratio,
        "risk_reward_label": rec.risk_reward_label,
        "rating_distribution": rec.analyst_data.rating_distribution if rec.analyst_data else {},
        "news_count": rec.news_data.article_count if rec.news_data else 0,
        "news_bullish": rec.news_data.bullish_count if rec.news_data else 0,
        "news_bearish": rec.news_data.bearish_count if rec.news_data else 0,
        "news_neutral": rec.news_data.neutral_count if rec.news_data else 0,
    }

    # Add options data if available
    if rec.options_data and rec.options_data.metrics:
        metrics = rec.options_data.metrics
        result.update({
            "options_signal": rec.options_data.signal_summary,
            "put_call_signal": rec.options_data.put_call_signal,
            "iv_signal": rec.options_data.iv_signal,
            "unusual_activity_signal": rec.options_data.unusual_activity_signal,
            "put_call_ratio": metrics.put_call_volume_ratio,
            "iv_skew": metrics.iv_skew,
            "atm_iv": metrics.atm_iv,
            "total_call_volume": metrics.total_call_volume,
            "total_put_volume": metrics.total_put_volume,
            "unusual_calls": metrics.unusual_calls[:3] if metrics.unusual_calls else [],
            "unusual_puts": metrics.unusual_puts[:3] if metrics.unusual_puts else [],
        })
    else:
        result.update({
            "options_signal": "N/A",
            "put_call_signal": "neutral",
            "iv_signal": "normal",
            "unusual_activity_signal": "none",
            "put_call_ratio": None,
            "iv_skew": None,
            "atm_iv": None,
            "total_call_volume": 0,
            "total_put_volume": 0,
            "unusual_calls": [],
            "unusual_puts": [],
        })

    return result


def get_rec_color(rec_type: str) -> str:
    """Get color for recommendation type."""
    colors = {
        "Strong Buy": "#00c853",
        "Buy": "#4caf50",
        "Hold": "#ffc107",
        "Sell": "#f44336",
        "Strong Sell": "#b71c1c",
    }
    return colors.get(rec_type, "#9e9e9e")


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("📈 Stock/Crypto Recommender")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏆 Top Picks", "🔍 Analyze Stock", "🎰 Polymarket", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Data Sources:**
        - Analyst ratings (Yahoo Finance)
        - Prediction markets (Polymarket)
        - News sentiment (RSS feeds)
        - Options flow (stocks only)
        """
    )

    return page


def render_top_picks():
    """Render the Top Picks page."""
    st.title("🏆 Top Picks")
    st.markdown("Dynamically discover and analyze stocks/crypto based on analyst ratings and market activity.")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        asset_type = st.selectbox("Asset Type", ["stock", "crypto", "both"])

    with col2:
        top_n = st.slider("Number of Picks", min_value=5, max_value=30, value=15)

    with col3:
        st.write("")  # Spacer

    # Fetch button
    if st.button("🔄 Discover & Analyze", type="primary"):
        with st.spinner("Discovering and analyzing..."):
            if asset_type == "both":
                stocks = get_top_picks("stock", top_n // 2)
                crypto = get_top_picks("crypto", top_n // 2)
                picks = sorted(stocks + crypto, key=lambda x: x["overall_score"], reverse=True)[:top_n]
            else:
                picks = get_top_picks(asset_type, top_n)

            st.session_state["picks"] = picks

    # Display results
    if "picks" in st.session_state and st.session_state["picks"]:
        picks = st.session_state["picks"]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyzed", len(picks))
        with col2:
            buy_count = sum(1 for p in picks if "Buy" in p["recommendation"])
            st.metric("Buy Ratings", buy_count)
        with col3:
            avg_score = sum(p["overall_score"] for p in picks) / len(picks)
            st.metric("Avg Score", f"{avg_score:.1%}")
        with col4:
            high_conf = sum(1 for p in picks if p["confidence"] == "high")
            st.metric("High Confidence", high_conf)

        st.markdown("---")

        # Table
        df = pd.DataFrame(picks)
        df["Score"] = df["overall_score"].apply(lambda x: f"{x:.1%}")
        df["Target"] = df["target_price"].apply(lambda x: f"${x:,.0f}" if x else "-")
        df["Upside"] = df["upside_potential"].apply(lambda x: f"{x:+.1f}%" if x is not None else "-")
        df["R/R"] = df.apply(
            lambda r: f"{r['risk_reward_ratio']:.1f} ({r['risk_reward_label']})" if r.get("risk_reward_ratio") else "-",
            axis=1
        )
        df["Analysts"] = df.apply(lambda r: f"{r['analyst_score']:.0%} ({r['analyst_count']})", axis=1)
        df["Options"] = df.apply(
            lambda r: f"{r['options_score']:.0%}" if r.get("options_confidence", "none") != "none" else "-",
            axis=1
        )

        display_df = df[["symbol", "recommendation", "Score", "Target", "Upside", "R/R", "Analysts", "Options", "confidence"]]
        display_df.columns = ["Symbol", "Rec", "Score", "Target", "Upside", "R/R", "Analysts", "Options", "Conf"]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Target": st.column_config.TextColumn("Target", width="small"),
                "Upside": st.column_config.TextColumn("Upside", width="small"),
                "R/R": st.column_config.TextColumn("R/R", width="medium"),
            },
        )

        # Score distribution chart
        st.subheader("Score Distribution")
        fig = px.bar(
            df,
            x="symbol",
            y="overall_score",
            color="recommendation",
            color_discrete_map={
                "Strong Buy": "#00c853",
                "Buy": "#4caf50",
                "Hold": "#ffc107",
                "Sell": "#f44336",
                "Strong Sell": "#b71c1c",
            },
            labels={"symbol": "Symbol", "overall_score": "Score", "recommendation": "Recommendation"},
        )
        fig.update_layout(yaxis_tickformat=".0%", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


def render_analyze():
    """Render the Analyze Stock page."""
    st.title("🔍 Analyze Stock/Crypto")
    st.markdown("Get detailed analysis for a specific symbol.")

    # Input
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL, NVDA, BTC").upper()

    with col2:
        asset_type = st.selectbox("Type", ["stock", "crypto"])

    with col3:
        st.write("")  # Spacer
        analyze_btn = st.button("🔍 Analyze", type="primary")

    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                data = analyze_symbol(symbol, asset_type)
                st.session_state["analysis"] = data
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")

    # Display analysis
    if "analysis" in st.session_state:
        data = st.session_state["analysis"]

        # Header with recommendation
        rec_color = get_rec_color(data["recommendation"])
        st.markdown(
            f"""
            <div style="background-color: {rec_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_color};">
                <h2 style="margin: 0;">{data['symbol']} ({data['asset_type'].upper()})</h2>
                <h3 style="color: {rec_color}; margin: 5px 0;">{data['recommendation']}</h3>
                <p style="font-size: 24px; margin: 0;">Overall Score: <strong>{data['overall_score']:.1%}</strong></p>
                <p style="margin: 0;">Confidence: {data['confidence'].upper()}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Score breakdown
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Analyst Score",
                f"{data['analyst_score']:.1%}",
                delta=f"{data['analyst_count']} analysts",
            )

        with col2:
            st.metric(
                "Betting Markets",
                f"{data['betting_score']:.1%}",
                delta=data["betting_confidence"],
            )

        with col3:
            st.metric(
                "News Sentiment",
                f"{data['news_score']:.1%}",
                delta=f"{data['news_count']} articles",
            )

        with col4:
            options_conf = data.get("options_confidence", "none")
            if options_conf != "none":
                st.metric(
                    "Options Flow",
                    f"{data['options_score']:.1%}",
                    delta=data.get("options_signal", "neutral"),
                )
            else:
                st.metric("Options Flow", "-", delta="N/A")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Score breakdown pie chart
            st.subheader("Score Breakdown")
            labels = ["Analysts", "Markets", "News"]
            values = [data["analyst_score"], data["betting_score"], data["news_score"]]
            colors = ["#2196f3", "#ff9800", "#4caf50"]

            # Add options if available
            if data.get("options_confidence", "none") != "none":
                labels.append("Options")
                values.append(data["options_score"])
                colors.append("#9c27b0")  # Purple for options

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        marker_colors=colors,
                    )
                ]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Analyst rating distribution
            if data["rating_distribution"]:
                st.subheader("Analyst Ratings")
                ratings = data["rating_distribution"]
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=list(ratings.keys()),
                            y=list(ratings.values()),
                            marker_color=["#00c853", "#4caf50", "#ffc107", "#f44336", "#b71c1c"][: len(ratings)],
                        )
                    ]
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Price & Risk/Reward Analysis
        if data["current_price"] and data["target_price"]:
            st.subheader("💰 Price & Risk/Reward Analysis")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['current_price']:,.2f}")
            with col2:
                st.metric("Target Price", f"${data['target_price']:,.2f}")
            with col3:
                upside = data.get("upside_potential")
                if upside is not None:
                    delta_color = "normal" if upside >= 0 else "inverse"
                    st.metric("Upside Potential", f"{upside:+.1f}%")
                else:
                    st.metric("Upside Potential", "-")
            with col4:
                rr = data.get("risk_reward_ratio")
                if rr is not None:
                    rr_label = data.get("risk_reward_label", "")
                    st.metric("Risk/Reward", f"{rr:.2f}", delta=rr_label)
                else:
                    st.metric("Risk/Reward", "-")

            # Risk/Reward breakdown
            downside = data.get("downside_risk")
            if downside is not None and upside is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Estimated Downside Risk:** {downside:.1f}%")
                    atm_iv = data.get("atm_iv")
                    if atm_iv:
                        st.markdown(f"_(based on ATM IV: {atm_iv:.1%})_")
                    else:
                        st.markdown("_(using default estimate)_")

                with col2:
                    rr = data.get("risk_reward_ratio")
                    if rr is not None:
                        if rr >= 2:
                            st.success(f"✅ Favorable R/R ratio ({rr:.2f} >= 2.0)")
                        elif rr >= 1:
                            st.warning(f"⚖️ Neutral R/R ratio ({rr:.2f})")
                        else:
                            st.error(f"⚠️ Unfavorable R/R ratio ({rr:.2f} < 1.0)")

        # Supporting/Risk factors
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Supporting Factors")
            for factor in data["supporting_factors"]:
                st.markdown(f"- {factor}")
            if not data["supporting_factors"]:
                st.markdown("_No supporting factors identified_")

        with col2:
            st.subheader("⚠️ Risk Factors")
            for risk in data["risk_factors"]:
                st.markdown(f"- {risk}")
            if not data["risk_factors"]:
                st.markdown("_No risk factors identified_")

        # News sentiment breakdown
        if data["news_count"] > 0:
            st.subheader("News Sentiment")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullish", data["news_bullish"])
            with col2:
                st.metric("Neutral", data["news_neutral"])
            with col3:
                st.metric("Bearish", data["news_bearish"])

        # Options flow details (stocks only)
        if data.get("options_confidence", "none") != "none":
            st.subheader("📊 Options Flow Analysis")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                pcr = data.get("put_call_ratio")
                if pcr is not None:
                    pcr_color = "green" if pcr < 0.7 else "red" if pcr > 1.0 else "normal"
                    st.metric("Put/Call Ratio", f"{pcr:.2f}")
                else:
                    st.metric("Put/Call Ratio", "-")

            with col2:
                st.metric("P/C Signal", data.get("put_call_signal", "neutral").title())

            with col3:
                iv_skew = data.get("iv_skew")
                if iv_skew is not None:
                    st.metric("IV Skew", f"{iv_skew:.3f}")
                else:
                    st.metric("IV Skew", "-")

            with col4:
                st.metric("IV Signal", data.get("iv_signal", "normal").replace("_", " ").title())

            # Volume breakdown
            call_vol = data.get("total_call_volume", 0)
            put_vol = data.get("total_put_volume", 0)
            if call_vol > 0 or put_vol > 0:
                st.markdown(f"**Volume:** {call_vol:,} calls / {put_vol:,} puts")

            # ATM IV
            atm_iv = data.get("atm_iv")
            if atm_iv is not None:
                st.markdown(f"**ATM Implied Volatility:** {atm_iv:.1%}")

            # Unusual activity
            unusual_calls = data.get("unusual_calls", [])
            unusual_puts = data.get("unusual_puts", [])

            if unusual_calls or unusual_puts:
                st.markdown("#### Unusual Options Activity")
                col1, col2 = st.columns(2)

                with col1:
                    if unusual_calls:
                        st.markdown("**🟢 Unusual Calls:**")
                        for item in unusual_calls:
                            st.markdown(
                                f"- ${item['strike']} strike: {item['volume']:,} vol / "
                                f"{item['open_interest']:,} OI ({item['volume_oi_ratio']:.1f}x)"
                            )

                with col2:
                    if unusual_puts:
                        st.markdown("**🔴 Unusual Puts:**")
                        for item in unusual_puts:
                            st.markdown(
                                f"- ${item['strike']} strike: {item['volume']:,} vol / "
                                f"{item['open_interest']:,} OI ({item['volume_oi_ratio']:.1f}x)"
                            )


def render_polymarket():
    """Render the Polymarket page."""
    st.title("🎰 Polymarket Prediction Markets")
    st.markdown("Browse prediction markets for stocks, crypto, and more.")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        market_type = st.selectbox("Market Type", ["top", "crypto", "stock"])

    with col2:
        limit = st.slider("Number of Markets", min_value=5, max_value=30, value=15)

    # Search
    search_query = st.text_input("🔍 Search Markets", placeholder="e.g., bitcoin, tesla, recession")

    if st.button("🔄 Fetch Markets", type="primary"):
        with st.spinner("Fetching markets..."):
            if search_query:
                markets = search_markets(search_query, limit)
            else:
                markets = get_polymarket_data(market_type, limit)
            st.session_state["markets"] = markets

    # Display markets
    if "markets" in st.session_state and st.session_state["markets"]:
        markets = st.session_state["markets"]

        st.markdown(f"**Found {len(markets)} markets**")

        for market in markets:
            prob = market["probability"]
            prob_color = "#4caf50" if prob >= 0.6 else "#f44336" if prob <= 0.4 else "#ffc107"

            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{market['title']}**")

                with col2:
                    st.markdown(
                        f"<span style='color: {prob_color}; font-size: 20px; font-weight: bold;'>{prob:.0%}</span>",
                        unsafe_allow_html=True,
                    )

                with col3:
                    if market["liquidity"]:
                        st.markdown(f"${market['liquidity']:,.0f}")

                st.markdown("---")


def render_settings():
    """Render the Settings page."""
    st.title("⚙️ Settings")
    st.markdown("View and configure recommendation settings.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analyst Settings")
        st.metric("Min Analysts", config.analyst_min_count)
        st.metric("Max Analysts", config.analyst_max_count)
        st.metric("Horizon", f"{config.horizon_months[0]}-{config.horizon_months[1]} months")

    with col2:
        st.subheader("Scoring Weights")

        # Weights visualization
        options_weight = config.get("weights", "options", default=0.15)
        weights = {
            "Analysts": config.analyst_weight,
            "Betting Markets": config.betting_weight,
            "News": config.news_weight,
            "Options": options_weight,
        }

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    hole=0.4,
                    marker_colors=["#2196f3", "#ff9800", "#4caf50", "#9c27b0"],
                )
            ]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Data Sources")
    st.markdown(
        """
        | Source | Description |
        |--------|-------------|
        | **Yahoo Finance** | Analyst ratings, price targets, stock data, options chains |
        | **Polymarket** | Prediction market probabilities |
        | **CoinGecko** | Cryptocurrency data |
        | **RSS Feeds** | News sentiment from Yahoo Finance, MarketWatch, etc. |
        | **Options Flow** | Put/call ratios, IV skew, unusual activity (stocks only) |
        """
    )


def main():
    """Main Streamlit app."""
    page = render_sidebar()

    if page == "🏆 Top Picks":
        render_top_picks()
    elif page == "🔍 Analyze Stock":
        render_analyze()
    elif page == "🎰 Polymarket":
        render_polymarket()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
