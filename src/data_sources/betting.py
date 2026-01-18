"""Betting market data fetcher for prediction market sentiment."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
from cachetools import TTLCache

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class BettingMarket:
    """Individual betting market data."""

    market_id: str
    title: str
    description: str
    source: str  # 'polymarket', 'metaculus', etc.
    probability: float  # 0-1 probability
    volume: float | None  # Trading volume in USD
    liquidity: float | None
    end_date: datetime | None
    url: str | None = None
    related_symbols: list[str] = field(default_factory=list)


@dataclass
class BettingMarketSentiment:
    """Aggregated betting market sentiment for an asset."""

    symbol: str
    asset_type: str
    markets: list[BettingMarket] = field(default_factory=list)
    average_bullish_probability: float = 0.5
    total_volume: float = 0.0
    market_count: int = 0

    @property
    def sentiment_score(self) -> float:
        """Calculate sentiment score (0-1) from betting markets.

        Weights markets by their volume for more accurate sentiment.
        """
        if not self.markets:
            return 0.5  # Neutral if no data

        # Volume-weighted average if volume data available
        markets_with_volume = [m for m in self.markets if m.volume]
        if markets_with_volume:
            total_vol = sum(m.volume for m in markets_with_volume)
            if total_vol > 0:
                weighted_sum = sum(
                    m.probability * m.volume for m in markets_with_volume
                )
                return weighted_sum / total_vol

        # Simple average otherwise
        return sum(m.probability for m in self.markets) / len(self.markets)

    @property
    def confidence_level(self) -> str:
        """Determine confidence level based on market data quality."""
        if self.total_volume > 1_000_000:
            return "high"
        elif self.total_volume > 100_000:
            return "medium"
        elif self.total_volume > 10_000:
            return "low"
        else:
            return "very_low"


class BettingMarketFetcher:
    """Fetches prediction market data from various sources."""

    def __init__(self) -> None:
        """Initialize the betting market fetcher."""
        self._cache: TTLCache = TTLCache(
            maxsize=500,
            ttl=config.get("cache", "ttl_minutes", default=30) * 60,
        )
        self._session: aiohttp.ClientSession | None = None
        self._polymarket_base = config.get(
            "betting_markets", "polymarket", "base_url",
            default="https://gamma-api.polymarket.com"
        )

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

    async def get_market_sentiment(
        self, symbol: str, asset_type: str = "stock"
    ) -> BettingMarketSentiment:
        """Get betting market sentiment for an asset.

        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTC')
            asset_type: 'stock' or 'crypto'

        Returns:
            BettingMarketSentiment with aggregated data
        """
        cache_key = f"betting_{asset_type}_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch from multiple sources
        markets: list[BettingMarket] = []

        # Polymarket
        if config.get("betting_markets", "polymarket", "enabled", default=True):
            polymarket_results = await self._fetch_polymarket_markets(
                symbol, asset_type
            )
            markets.extend(polymarket_results)

        # Calculate aggregate sentiment
        sentiment = BettingMarketSentiment(
            symbol=symbol,
            asset_type=asset_type,
            markets=markets,
            average_bullish_probability=(
                sum(m.probability for m in markets) / len(markets)
                if markets
                else 0.5
            ),
            total_volume=sum(m.volume or 0 for m in markets),
            market_count=len(markets),
        )

        self._cache[cache_key] = sentiment
        return sentiment

    async def _fetch_polymarket_markets(
        self, symbol: str, asset_type: str
    ) -> list[BettingMarket]:
        """Fetch relevant markets from Polymarket.

        Polymarket API provides prediction markets. We search for markets
        related to the given symbol and extract sentiment.
        """
        session = await self._get_session()
        markets: list[BettingMarket] = []

        # Build search queries based on asset type
        search_terms = self._build_search_terms(symbol, asset_type)

        try:
            # Polymarket CLOB API for markets - fetch active markets only
            url = f"{self._polymarket_base}/markets"
            params = {"closed": "false", "limit": "500"}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Filter markets relevant to our symbol
                    for market in data if isinstance(data, list) else []:
                        if self._is_market_relevant(market, search_terms):
                            parsed = self._parse_polymarket_market(market, symbol)
                            if parsed:
                                markets.append(parsed)

        except aiohttp.ClientError as e:
            logger.warning(f"Error fetching Polymarket data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with Polymarket: {e}")

        return markets

    def _build_search_terms(self, symbol: str, asset_type: str) -> list[str]:
        """Build search terms for finding relevant markets."""
        terms = [symbol.upper(), symbol.lower()]

        # Add full names for common assets
        stock_names = {
            "AAPL": ["apple", "iphone"],
            "GOOGL": ["google", "alphabet"],
            "MSFT": ["microsoft", "windows"],
            "AMZN": ["amazon", "aws"],
            "TSLA": ["tesla", "elon musk"],
            "META": ["meta", "facebook", "instagram"],
            "NVDA": ["nvidia", "gpu"],
            "AMD": ["amd", "advanced micro"],
        }

        crypto_names = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "ether"],
            "SOL": ["solana"],
            "XRP": ["ripple", "xrp"],
            "DOGE": ["dogecoin", "doge"],
            "ADA": ["cardano"],
        }

        name_map = crypto_names if asset_type == "crypto" else stock_names
        if symbol.upper() in name_map:
            terms.extend(name_map[symbol.upper()])

        return terms

    def _is_market_relevant(
        self, market: dict[str, Any], search_terms: list[str]
    ) -> bool:
        """Check if a market is relevant to our search terms."""
        title = market.get("question", "").lower()
        description = market.get("description", "").lower()
        combined = f"{title} {description}"

        # Check for price-related markets
        price_keywords = ["price", "reach", "hit", "above", "below", "ath", "all-time"]

        has_price_keyword = any(kw in combined for kw in price_keywords)
        has_search_term = any(term.lower() in combined for term in search_terms)

        return has_price_keyword and has_search_term

    def _parse_polymarket_market(
        self, market: dict[str, Any], symbol: str
    ) -> BettingMarket | None:
        """Parse a Polymarket market into our data structure."""
        try:
            import json as json_module

            market_id = str(market.get("id") or market.get("conditionId", ""))
            title = market.get("question", "")
            description = market.get("description", "")

            # Get outcome prices (probability)
            # outcomePrices is a JSON string like "[\"0.029\", \"0.971\"]"
            probability = 0.5
            outcome_prices = market.get("outcomePrices")
            if outcome_prices:
                try:
                    if isinstance(outcome_prices, str):
                        prices = json_module.loads(outcome_prices)
                    else:
                        prices = outcome_prices
                    if prices and len(prices) > 0:
                        # First price is "Yes" probability
                        probability = float(prices[0])
                except (json_module.JSONDecodeError, ValueError, IndexError):
                    pass

            # Fallback to tokens if available
            if probability == 0.5:
                tokens = market.get("tokens", [])
                if tokens and isinstance(tokens, list) and len(tokens) > 0:
                    yes_token = tokens[0]
                    if isinstance(yes_token, dict):
                        probability = float(yes_token.get("price", 0.5))

            # Determine if this is a bullish or bearish market
            is_bullish = self._is_bullish_market(title)
            if not is_bullish:
                probability = 1 - probability  # Invert for bearish markets

            volume = float(market.get("volume", 0) or market.get("volumeNum", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or market.get("liquidityNum", 0) or 0)

            end_date = None
            date_str = market.get("endDate") or market.get("end_date_iso")
            if date_str:
                try:
                    end_date = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

            return BettingMarket(
                market_id=market_id,
                title=title,
                description=description[:500] if description else "",
                source="polymarket",
                probability=probability,
                volume=volume,
                liquidity=liquidity,
                end_date=end_date,
                url=f"https://polymarket.com/event/{market.get('slug', market_id)}",
                related_symbols=[symbol],
            )

        except Exception as e:
            logger.debug(f"Error parsing Polymarket market: {e}")
            return None

    def _is_bullish_market(self, title: str) -> bool:
        """Determine if a market is asking about bullish outcome.

        Examples of bullish: "Will X reach $100?" "Will X hit ATH?"
        Examples of bearish: "Will X drop below $50?" "Will X crash?"
        """
        title_lower = title.lower()

        bullish_patterns = [
            r"reach\s+\$?\d",
            r"hit\s+\$?\d",
            r"above\s+\$?\d",
            r"surpass",
            r"all.time.high",
            r"ath",
            r"rise",
            r"increase",
            r"go up",
            r"moon",
        ]

        bearish_patterns = [
            r"drop\s+(below|to)",
            r"fall\s+(below|to)",
            r"crash",
            r"below\s+\$?\d",
            r"decline",
            r"decrease",
        ]

        for pattern in bullish_patterns:
            if re.search(pattern, title_lower):
                return True

        for pattern in bearish_patterns:
            if re.search(pattern, title_lower):
                return False

        # Default to bullish interpretation
        return True

    async def get_general_market_sentiment(self) -> dict[str, Any]:
        """Get general market sentiment from prediction markets.

        Returns sentiment on broader market conditions (recession, Fed rates, etc.)
        """
        session = await self._get_session()
        sentiment_data: dict[str, Any] = {
            "recession_probability": None,
            "rate_cut_probability": None,
            "bull_market_probability": None,
            "markets": [],
        }

        try:
            url = f"{self._polymarket_base}/markets"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    macro_keywords = [
                        "recession",
                        "fed",
                        "interest rate",
                        "inflation",
                        "bull market",
                        "bear market",
                        "s&p 500",
                        "stock market",
                    ]

                    for market in data if isinstance(data, list) else []:
                        title = market.get("question", "").lower()
                        if any(kw in title for kw in macro_keywords):
                            parsed = self._parse_polymarket_market(market, "MACRO")
                            if parsed:
                                sentiment_data["markets"].append(parsed)

                                # Extract specific probabilities
                                if "recession" in title:
                                    sentiment_data["recession_probability"] = (
                                        parsed.probability
                                    )
                                elif "rate cut" in title or "fed" in title:
                                    sentiment_data["rate_cut_probability"] = (
                                        parsed.probability
                                    )

        except Exception as e:
            logger.warning(f"Error fetching general market sentiment: {e}")

        return sentiment_data

    async def search_markets(
        self, query: str, limit: int = 10
    ) -> list[BettingMarket]:
        """Search for prediction markets matching a query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching BettingMarket objects
        """
        session = await self._get_session()
        markets: list[BettingMarket] = []

        try:
            url = f"{self._polymarket_base}/markets"
            params = {"closed": "false", "limit": "500"}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    query_lower = query.lower()

                    for market in data if isinstance(data, list) else []:
                        title = market.get("question", "").lower()
                        desc = market.get("description", "").lower()

                        if query_lower in title or query_lower in desc:
                            parsed = self._parse_polymarket_market(market, query)
                            if parsed:
                                markets.append(parsed)

                    # Sort by liquidity and return top results
                    markets.sort(key=lambda m: m.liquidity or 0, reverse=True)
                    markets = markets[:limit]

        except Exception as e:
            logger.warning(f"Error searching markets: {e}")

        return markets

    async def get_crypto_markets(self, limit: int = 20) -> list[BettingMarket]:
        """Get all crypto-related prediction markets.

        Returns:
            List of crypto-related BettingMarket objects sorted by liquidity
        """
        session = await self._get_session()
        markets: list[BettingMarket] = []

        # Strong crypto keywords - use word boundary matching
        # These patterns need to match as whole words to avoid "eth" matching "Netherlands"
        strong_crypto_patterns = [
            r"\bbitcoin\b", r"\bethereum\b", r"\bbtc\b", r"\beth\b",
            r"\bsolana\b", r"\bdogecoin\b", r"\bripple\b", r"\bxrp\b",
            r"\bcardano\b", r"\bcrypto\b", r"\bcryptocurrency\b",
            r"\bmegaeth\b",
        ]

        # Weaker keywords that might appear in description
        weak_crypto_patterns = [
            r"\bblockchain\b", r"\baltcoin\b", r"\bdefi\b",
        ]

        try:
            url = f"{self._polymarket_base}/markets"
            params = {"closed": "false", "limit": "500"}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    for market in data if isinstance(data, list) else []:
                        title = market.get("question", "").lower()
                        desc = market.get("description", "").lower()
                        combined = f"{title} {desc}"

                        # Strong match: crypto keyword in title = definitely crypto
                        has_strong_crypto = any(
                            re.search(pattern, title) for pattern in strong_crypto_patterns
                        )

                        # Weak match: crypto keyword in description
                        has_weak_crypto = any(
                            re.search(pattern, combined) for pattern in weak_crypto_patterns
                        )

                        if has_strong_crypto or has_weak_crypto:
                            parsed = self._parse_polymarket_market(market, "CRYPTO")
                            if parsed:
                                markets.append(parsed)

                    # Sort by liquidity
                    markets.sort(key=lambda m: m.liquidity or 0, reverse=True)
                    markets = markets[:limit]

        except Exception as e:
            logger.warning(f"Error fetching crypto markets: {e}")

        return markets

    async def get_stock_markets(self, limit: int = 20) -> list[BettingMarket]:
        """Get all stock/market-related prediction markets.

        Returns:
            List of stock-related BettingMarket objects sorted by liquidity
        """
        session = await self._get_session()
        markets: list[BettingMarket] = []

        # Financial/stock market patterns with word boundaries
        stock_patterns = [
            r"stock market", r"stock price", r"s&p 500", r"s&p500", r"\bnasdaq\b",
            r"dow jones", r"\bfed\b rate", r"interest rate", r"\brecession\b",
            r"\binflation\b", r"\bgdp\b", r"tesla stock", r"apple stock",
            r"nvidia stock", r"amazon stock", r"microsoft stock", r"google stock",
            r"\bearnings\b", r"\bipo\b", r"market crash", r"bull market", r"bear market",
        ]

        try:
            url = f"{self._polymarket_base}/markets"
            params = {"closed": "false", "limit": "500"}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    for market in data if isinstance(data, list) else []:
                        title = market.get("question", "").lower()
                        desc = market.get("description", "").lower()
                        combined = f"{title} {desc}"

                        # Must have a stock/finance keyword
                        has_stock = any(
                            re.search(pattern, combined) for pattern in stock_patterns
                        )

                        if has_stock:
                            parsed = self._parse_polymarket_market(market, "STOCK")
                            if parsed:
                                markets.append(parsed)

                    # Sort by liquidity
                    markets.sort(key=lambda m: m.liquidity or 0, reverse=True)
                    markets = markets[:limit]

        except Exception as e:
            logger.warning(f"Error fetching stock markets: {e}")

        return markets

    async def get_top_markets(self, limit: int = 20) -> list[BettingMarket]:
        """Get top prediction markets by liquidity.

        Returns:
            List of top BettingMarket objects sorted by liquidity
        """
        session = await self._get_session()
        markets: list[BettingMarket] = []

        try:
            url = f"{self._polymarket_base}/markets"
            params = {"closed": "false", "limit": "200"}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    for market in data if isinstance(data, list) else []:
                        parsed = self._parse_polymarket_market(market, "ALL")
                        if parsed:
                            markets.append(parsed)

                    # Sort by liquidity
                    markets.sort(key=lambda m: m.liquidity or 0, reverse=True)
                    markets = markets[:limit]

        except Exception as e:
            logger.warning(f"Error fetching top markets: {e}")

        return markets
