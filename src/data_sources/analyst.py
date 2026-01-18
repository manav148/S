"""Analyst data fetcher for stock and crypto recommendations."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
from cachetools import TTLCache

from ..config import config

logger = logging.getLogger(__name__)


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

        consensus = await self._fetch_yahoo_finance_data(symbol)
        self._cache[cache_key] = consensus
        return consensus

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

    async def _fetch_yahoo_finance_data(self, symbol: str) -> AnalystConsensus:
        """Fetch analyst data from Yahoo Finance API.

        Uses the unofficial Yahoo Finance API endpoints to get analyst recommendations.
        """
        session = await self._get_session()
        recommendations: list[AnalystRecommendation] = []
        current_price = None
        average_target = None

        try:
            # Yahoo Finance API for quote data
            quote_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
            params = {"modules": "recommendationTrend,financialData,price"}

            async with session.get(quote_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("quoteSummary", {}).get("result", [])

                    if result:
                        result = result[0]

                        # Get current price
                        price_data = result.get("price", {})
                        current_price = price_data.get("regularMarketPrice", {}).get(
                            "raw"
                        )

                        # Get target price
                        financial_data = result.get("financialData", {})
                        average_target = financial_data.get(
                            "targetMeanPrice", {}
                        ).get("raw")

                        # Get recommendation trend
                        trend_data = result.get("recommendationTrend", {}).get(
                            "trend", []
                        )

                        # Process recommendations from trend data
                        recommendations = self._parse_yahoo_recommendations(
                            trend_data, current_price, average_target
                        )

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")

        return AnalystConsensus(
            symbol=symbol,
            asset_type="stock",
            recommendations=recommendations,
            current_price=current_price,
            average_target_price=average_target,
            consensus_rating=self._calculate_consensus_rating(recommendations),
            total_analysts=len(recommendations),
        )

    def _parse_yahoo_recommendations(
        self,
        trend_data: list[dict[str, Any]],
        current_price: float | None,
        target_price: float | None,
    ) -> list[AnalystRecommendation]:
        """Parse Yahoo Finance recommendation trend data into individual recommendations."""
        recommendations: list[AnalystRecommendation] = []

        # Use the most recent period (first item usually)
        if not trend_data:
            return recommendations

        period = trend_data[0]
        period_date = period.get("period", "0m")

        # Calculate upside if we have prices
        upside = None
        if current_price and target_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100

        # Create synthetic recommendations based on counts
        rating_map = {
            "strongBuy": Rating.STRONG_BUY,
            "buy": Rating.BUY,
            "hold": Rating.HOLD,
            "sell": Rating.SELL,
            "strongSell": Rating.STRONG_SELL,
        }

        analyst_num = 1
        for rating_key, rating_value in rating_map.items():
            count = period.get(rating_key, 0)
            for i in range(count):
                recommendations.append(
                    AnalystRecommendation(
                        analyst_name=f"Analyst_{analyst_num}",
                        firm=f"Firm_{analyst_num}",
                        rating=rating_value,
                        target_price=target_price,
                        current_price=current_price,
                        upside_percent=upside,
                        date=datetime.now(),
                        horizon_months=12,
                    )
                )
                analyst_num += 1

        return recommendations

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
