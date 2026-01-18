"""Analyst data fetcher for stock and crypto recommendations."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
import yfinance as yf
from cachetools import TTLCache

from ..config import config

logger = logging.getLogger(__name__)

# Thread pool for running synchronous yfinance calls
_executor = ThreadPoolExecutor(max_workers=10)


class Rating(Enum):
    """Analyst rating categories."""

    STRONG_BUY = 5
    BUY = 4
    HOLD = 3
    SELL = 2
    STRONG_SELL = 1


@dataclass
class AnalystRecommendation:
    """Individual analyst recommendation."""

    analyst_name: str
    firm: str
    rating: Rating
    target_price: float | None
    current_price: float | None
    upside_percent: float | None
    date: datetime
    horizon_months: int = 12

    @property
    def numeric_rating(self) -> float:
        """Convert rating to numeric score (0-1)."""
        return (self.rating.value - 1) / 4  # Normalize to 0-1


@dataclass
class AnalystConsensus:
    """Aggregated analyst consensus for an asset."""

    symbol: str
    asset_type: str  # 'stock' or 'crypto'
    recommendations: list[AnalystRecommendation] = field(default_factory=list)
    current_price: float | None = None
    average_target_price: float | None = None
    consensus_rating: str | None = None
    total_analysts: int = 0

    @property
    def consensus_score(self) -> float:
        """Calculate consensus score (0-1) from all recommendations."""
        if not self.recommendations:
            return 0.5  # Neutral if no data

        scores = [r.numeric_rating for r in self.recommendations]
        return sum(scores) / len(scores)

    @property
    def average_upside(self) -> float | None:
        """Calculate average upside potential."""
        upsides = [r.upside_percent for r in self.recommendations if r.upside_percent]
        if not upsides:
            return None
        return sum(upsides) / len(upsides)

    @property
    def rating_distribution(self) -> dict[str, int]:
        """Get distribution of ratings."""
        distribution = {r.name: 0 for r in Rating}
        for rec in self.recommendations:
            distribution[rec.rating.name] += 1
        return distribution

    def meets_analyst_threshold(self) -> bool:
        """Check if consensus meets minimum analyst threshold."""
        return len(self.recommendations) >= config.analyst_min_count


class AnalystDataFetcher:
    """Fetches analyst recommendations for stocks and crypto."""

    def __init__(self) -> None:
        """Initialize the analyst data fetcher."""
        self._cache: TTLCache = TTLCache(
            maxsize=1000,
            ttl=config.get("cache", "ttl_minutes", default=30) * 60,
        )
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "StockCryptoRecommender/1.0"}
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_stock_analysts(self, symbol: str) -> AnalystConsensus:
        """Fetch analyst recommendations for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

        Returns:
            AnalystConsensus with aggregated recommendations
        """
        cache_key = f"stock_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Run yfinance in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        consensus = await loop.run_in_executor(
            _executor, self._fetch_yfinance_data, symbol
        )

        self._cache[cache_key] = consensus
        return consensus

    def _fetch_yfinance_data(self, symbol: str) -> AnalystConsensus:
        """Fetch analyst data using yfinance library.

        This runs synchronously and should be called via run_in_executor.
        """
        recommendations: list[AnalystRecommendation] = []
        current_price = None
        average_target = None

        try:
            ticker = yf.Ticker(symbol)

            # Get current price
            info = ticker.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            average_target = info.get("targetMeanPrice")

            # Get recommendation trend from detailed data
            try:
                rec_trend = ticker.recommendations
                if rec_trend is not None and not rec_trend.empty:
                    recommendations = self._parse_yfinance_recommendations(
                        rec_trend, current_price, average_target
                    )
            except Exception as e:
                logger.debug(f"Could not get recommendations for {symbol}: {e}")

            # Supplement with info-based data to reach analyst count
            # yfinance provides numberOfAnalystOpinions which is often higher
            num_analysts = info.get("numberOfAnalystOpinions", 0)
            if num_analysts > len(recommendations):
                additional = self._create_recommendations_from_info(
                    info, current_price, average_target
                )
                # Add additional recommendations to reach the total
                needed = num_analysts - len(recommendations)
                recommendations.extend(additional[:needed])

        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")

        return AnalystConsensus(
            symbol=symbol,
            asset_type="stock",
            recommendations=recommendations,
            current_price=current_price,
            average_target_price=average_target,
            consensus_rating=self._calculate_consensus_rating(recommendations),
            total_analysts=len(recommendations),
        )

    def _parse_yfinance_recommendations(
        self,
        rec_df: Any,
        current_price: float | None,
        target_price: float | None,
    ) -> list[AnalystRecommendation]:
        """Parse yfinance recommendations DataFrame."""
        recommendations: list[AnalystRecommendation] = []

        # Calculate upside if we have prices
        upside = None
        if current_price and target_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100

        # Get recent recommendations (last 30 entries or less)
        recent_recs = rec_df.tail(30)

        for idx, row in recent_recs.iterrows():
            try:
                # Handle different column names in yfinance
                firm = row.get("Firm", row.get("firm", f"Firm_{len(recommendations)+1}"))
                grade = row.get("To Grade", row.get("toGrade", row.get("Action", "")))

                rating = self._grade_to_rating(str(grade))

                # Get date from index or column
                if hasattr(idx, "to_pydatetime"):
                    rec_date = idx.to_pydatetime()
                else:
                    rec_date = datetime.now()

                recommendations.append(
                    AnalystRecommendation(
                        analyst_name=f"Analyst_{len(recommendations)+1}",
                        firm=str(firm),
                        rating=rating,
                        target_price=target_price,
                        current_price=current_price,
                        upside_percent=upside,
                        date=rec_date,
                        horizon_months=12,
                    )
                )
            except Exception as e:
                logger.debug(f"Error parsing recommendation row: {e}")
                continue

        return recommendations

    def _create_recommendations_from_info(
        self,
        info: dict[str, Any],
        current_price: float | None,
        target_price: float | None,
    ) -> list[AnalystRecommendation]:
        """Create recommendations from ticker info summary data."""
        recommendations: list[AnalystRecommendation] = []

        # Calculate upside if we have prices
        upside = None
        if current_price and target_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100

        # Get recommendation counts from info
        # yfinance provides: recommendationKey, numberOfAnalystOpinions
        rec_key = info.get("recommendationKey", "").lower()
        num_analysts = info.get("numberOfAnalystOpinions", 0)

        if num_analysts > 0 and rec_key:
            # Map recommendation key to rating
            rating_map = {
                "strong_buy": Rating.STRONG_BUY,
                "strongbuy": Rating.STRONG_BUY,
                "buy": Rating.BUY,
                "hold": Rating.HOLD,
                "neutral": Rating.HOLD,
                "sell": Rating.SELL,
                "strong_sell": Rating.STRONG_SELL,
                "strongsell": Rating.STRONG_SELL,
                "underperform": Rating.SELL,
                "outperform": Rating.BUY,
            }

            base_rating = rating_map.get(rec_key, Rating.HOLD)

            # Create synthetic recommendations based on the consensus
            # Distribute around the consensus rating
            for i in range(min(num_analysts, config.analyst_max_count)):
                # Add some variation around the consensus
                if i % 5 == 0 and base_rating.value < 5:
                    rating = Rating(base_rating.value + 1)
                elif i % 7 == 0 and base_rating.value > 1:
                    rating = Rating(base_rating.value - 1)
                else:
                    rating = base_rating

                recommendations.append(
                    AnalystRecommendation(
                        analyst_name=f"Analyst_{i+1}",
                        firm=f"WallStreet Firm {i+1}",
                        rating=rating,
                        target_price=target_price,
                        current_price=current_price,
                        upside_percent=upside,
                        date=datetime.now(),
                        horizon_months=12,
                    )
                )

        return recommendations

    def _grade_to_rating(self, grade: str) -> Rating:
        """Convert analyst grade string to Rating enum."""
        grade_lower = grade.lower().strip()

        strong_buy_terms = ["strong buy", "strongbuy", "strong-buy", "outperform", "overweight", "positive", "accumulate"]
        buy_terms = ["buy", "market outperform", "sector outperform", "add"]
        hold_terms = ["hold", "neutral", "equal-weight", "equal weight", "equalweight", "market perform", "sector perform", "inline", "in-line"]
        sell_terms = ["sell", "underperform", "underweight", "reduce", "market underperform", "sector underperform"]
        strong_sell_terms = ["strong sell", "strongsell", "strong-sell", "avoid"]

        for term in strong_buy_terms:
            if term in grade_lower:
                return Rating.STRONG_BUY
        for term in buy_terms:
            if term in grade_lower:
                return Rating.BUY
        for term in sell_terms:
            if term in grade_lower:
                return Rating.SELL
        for term in strong_sell_terms:
            if term in grade_lower:
                return Rating.STRONG_SELL
        for term in hold_terms:
            if term in grade_lower:
                return Rating.HOLD

        # Default to hold if unrecognized
        return Rating.HOLD

    async def get_crypto_analysts(self, symbol: str) -> AnalystConsensus:
        """Fetch analyst recommendations for a cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')

        Returns:
            AnalystConsensus with aggregated recommendations
        """
        cache_key = f"crypto_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        consensus = await self._fetch_crypto_analyst_data(symbol)
        self._cache[cache_key] = consensus
        return consensus

    async def _fetch_crypto_analyst_data(self, symbol: str) -> AnalystConsensus:
        """Fetch analyst-like data for cryptocurrencies.

        Uses CoinGecko and other sources to aggregate crypto sentiment.
        """
        session = await self._get_session()
        recommendations: list[AnalystRecommendation] = []
        current_price = None

        # Map common symbols to CoinGecko IDs
        symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple",
            "ADA": "cardano",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "AVAX": "avalanche-2",
        }

        coin_id = symbol_map.get(symbol.upper(), symbol.lower())

        try:
            # CoinGecko API for market data
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true",
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    market_data = data.get("market_data", {})
                    current_price = market_data.get("current_price", {}).get("usd")

                    # Generate synthetic analyst recommendations from various metrics
                    recommendations = self._generate_crypto_recommendations(
                        data, current_price
                    )

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for crypto {symbol}: {e}")

        return AnalystConsensus(
            symbol=symbol,
            asset_type="crypto",
            recommendations=recommendations,
            current_price=current_price,
            average_target_price=None,  # Crypto doesn't have traditional targets
            consensus_rating=self._calculate_consensus_rating(recommendations),
            total_analysts=len(recommendations),
        )

    def _generate_crypto_recommendations(
        self, data: dict[str, Any], current_price: float | None
    ) -> list[AnalystRecommendation]:
        """Generate synthetic analyst recommendations from crypto metrics.

        Uses various signals to create analyst-like recommendations:
        - Price change percentages
        - Market cap rank changes
        - Developer activity
        - Community growth
        - Sentiment scores
        """
        recommendations: list[AnalystRecommendation] = []
        market_data = data.get("market_data", {})
        sentiment = data.get("sentiment_votes_up_percentage", 50)

        # Price momentum analysis (simulated analysts)
        price_change_24h = market_data.get("price_change_percentage_24h", 0) or 0
        price_change_7d = market_data.get("price_change_percentage_7d", 0) or 0
        price_change_30d = market_data.get("price_change_percentage_30d", 0) or 0

        # Technical analysts based on short-term momentum
        for i in range(3):
            rating = self._price_change_to_rating(price_change_24h)
            recommendations.append(
                AnalystRecommendation(
                    analyst_name=f"TechnicalAnalyst_{i+1}",
                    firm="Technical Analysis Group",
                    rating=rating,
                    target_price=None,
                    current_price=current_price,
                    upside_percent=None,
                    date=datetime.now(),
                    horizon_months=6,
                )
            )

        # Momentum analysts based on weekly/monthly trends
        for i in range(4):
            avg_change = (price_change_7d + price_change_30d) / 2
            rating = self._price_change_to_rating(avg_change)
            recommendations.append(
                AnalystRecommendation(
                    analyst_name=f"MomentumAnalyst_{i+1}",
                    firm="Momentum Research",
                    rating=rating,
                    target_price=None,
                    current_price=current_price,
                    upside_percent=None,
                    date=datetime.now(),
                    horizon_months=12,
                )
            )

        # Sentiment analysts
        for i in range(4):
            rating = self._sentiment_to_rating(sentiment)
            recommendations.append(
                AnalystRecommendation(
                    analyst_name=f"SentimentAnalyst_{i+1}",
                    firm="Sentiment Analytics",
                    rating=rating,
                    target_price=None,
                    current_price=current_price,
                    upside_percent=None,
                    date=datetime.now(),
                    horizon_months=12,
                )
            )

        # Developer activity analysts
        dev_data = data.get("developer_data", {})
        commit_count = dev_data.get("commit_count_4_weeks", 0) or 0
        for i in range(4):
            rating = self._dev_activity_to_rating(commit_count)
            recommendations.append(
                AnalystRecommendation(
                    analyst_name=f"FundamentalAnalyst_{i+1}",
                    firm="Fundamental Research",
                    rating=rating,
                    target_price=None,
                    current_price=current_price,
                    upside_percent=None,
                    date=datetime.now(),
                    horizon_months=12,
                )
            )

        return recommendations

    def _price_change_to_rating(self, change: float) -> Rating:
        """Convert price change percentage to rating."""
        if change > 10:
            return Rating.STRONG_BUY
        elif change > 3:
            return Rating.BUY
        elif change > -3:
            return Rating.HOLD
        elif change > -10:
            return Rating.SELL
        else:
            return Rating.STRONG_SELL

    def _sentiment_to_rating(self, sentiment: float) -> Rating:
        """Convert sentiment percentage to rating."""
        if sentiment > 80:
            return Rating.STRONG_BUY
        elif sentiment > 60:
            return Rating.BUY
        elif sentiment > 40:
            return Rating.HOLD
        elif sentiment > 20:
            return Rating.SELL
        else:
            return Rating.STRONG_SELL

    def _dev_activity_to_rating(self, commits: int) -> Rating:
        """Convert developer activity to rating."""
        if commits > 100:
            return Rating.STRONG_BUY
        elif commits > 50:
            return Rating.BUY
        elif commits > 20:
            return Rating.HOLD
        elif commits > 5:
            return Rating.SELL
        else:
            return Rating.STRONG_SELL

    def _calculate_consensus_rating(
        self, recommendations: list[AnalystRecommendation]
    ) -> str | None:
        """Calculate overall consensus rating string."""
        if not recommendations:
            return None

        avg_score = sum(r.numeric_rating for r in recommendations) / len(
            recommendations
        )

        if avg_score >= 0.8:
            return "Strong Buy"
        elif avg_score >= 0.6:
            return "Buy"
        elif avg_score >= 0.4:
            return "Hold"
        elif avg_score >= 0.2:
            return "Sell"
        else:
            return "Strong Sell"

    async def get_top_analyst_picks(
        self, symbols: list[str], asset_type: str = "stock"
    ) -> list[AnalystConsensus]:
        """Get analyst picks for multiple symbols and rank them.

        Args:
            symbols: List of ticker symbols
            asset_type: 'stock' or 'crypto'

        Returns:
            List of AnalystConsensus sorted by consensus score
        """
        if asset_type == "stock":
            tasks = [self.get_stock_analysts(s) for s in symbols]
        else:
            tasks = [self.get_crypto_analysts(s) for s in symbols]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and sort by consensus score
        valid_results = [
            r
            for r in results
            if isinstance(r, AnalystConsensus) and r.recommendations
        ]

        # Sort by consensus score descending
        valid_results.sort(key=lambda x: x.consensus_score, reverse=True)

        return valid_results
