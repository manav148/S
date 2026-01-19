"""Dynamic stock screener for discovering stocks to analyze."""

import asyncio
import logging
from dataclasses import dataclass

import aiohttp
from cachetools import TTLCache

logger = logging.getLogger(__name__)


@dataclass
class ScreenerResult:
    """Result from stock screener."""

    symbol: str
    name: str
    source: str  # 'analyst', 'active', 'trending'
    score: float  # Relevance score for ranking


class StockScreener:
    """Screens and discovers stocks dynamically based on market data."""

    # Yahoo Finance screener API endpoint
    SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

    def __init__(self) -> None:
        """Initialize the screener."""
        self._cache: TTLCache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                }
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_analyst_top_picks(self, limit: int = 25) -> list[str]:
        """Get stocks with strong analyst buy ratings.

        Uses Yahoo Finance screener to find stocks with strong buy consensus.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of stock symbols
        """
        cache_key = f"analyst_picks_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        symbols = await self._fetch_screener("undervalued_growth_stocks", limit)
        if not symbols:
            symbols = await self._fetch_screener("growth_technology_stocks", limit)
        if not symbols:
            symbols = self._get_fallback_stocks()[:limit]

        self._cache[cache_key] = symbols
        return symbols

    async def _fetch_screener(self, screener_type: str, limit: int) -> list[str]:
        """Fetch stocks from Yahoo Finance screener API.

        Args:
            screener_type: Type of screener (most_actives, day_gainers, etc.)
            limit: Maximum number of results

        Returns:
            List of stock symbols
        """
        session = await self._get_session()

        try:
            params = {
                "scrIds": screener_type,
                "count": limit,
            }

            async with session.get(self.SCREENER_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quotes = (
                        data.get("finance", {})
                        .get("result", [{}])[0]
                        .get("quotes", [])
                    )
                    symbols = []
                    for quote in quotes[:limit]:
                        symbol = quote.get("symbol", "")
                        if symbol and self._is_valid_symbol(symbol):
                            symbols.append(symbol)
                    return symbols

        except Exception as e:
            logger.debug(f"Screener {screener_type} failed: {e}")

        return []

    async def get_most_active(self, limit: int = 25) -> list[str]:
        """Get most actively traded stocks.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of stock symbols
        """
        cache_key = f"most_active_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        symbols = await self._fetch_screener("most_actives", limit)
        self._cache[cache_key] = symbols
        return symbols

    async def get_day_gainers(self, limit: int = 15) -> list[str]:
        """Get top gaining stocks today.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of stock symbols
        """
        cache_key = f"day_gainers_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        symbols = await self._fetch_screener("day_gainers", limit)
        self._cache[cache_key] = symbols
        return symbols

    async def discover_stocks(self, limit: int = 30) -> list[str]:
        """Discover stocks using multiple sources.

        Combines analyst picks, most active, and trending stocks.

        Args:
            limit: Maximum total stocks to return

        Returns:
            List of unique stock symbols
        """
        cache_key = f"discover_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch from multiple sources concurrently
        analyst_task = self.get_analyst_top_picks(limit // 2)
        active_task = self.get_most_active(limit // 2)
        gainers_task = self.get_day_gainers(limit // 4)

        analyst_picks, most_active, gainers = await asyncio.gather(
            analyst_task, active_task, gainers_task,
            return_exceptions=True
        )

        # Combine results, maintaining order priority
        seen = set()
        combined = []

        # Add from each source, prioritizing analyst picks
        for source in [analyst_picks, most_active, gainers]:
            if isinstance(source, Exception):
                continue
            for symbol in source:
                if symbol not in seen and self._is_valid_symbol(symbol):
                    seen.add(symbol)
                    combined.append(symbol)

        # If we still don't have enough, add fallback stocks
        if len(combined) < limit:
            for symbol in self._get_fallback_stocks():
                if symbol not in seen:
                    seen.add(symbol)
                    combined.append(symbol)
                if len(combined) >= limit:
                    break

        result = combined[:limit]
        self._cache[cache_key] = result
        return result

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid for our analysis."""
        if not symbol:
            return False
        # Filter out non-standard symbols (warrants, preferred, etc.)
        if any(c in symbol for c in ['-', '.', '^']):
            return False
        # Filter out very short or very long symbols
        if len(symbol) < 1 or len(symbol) > 5:
            return False
        return True

    def _get_fallback_stocks(self) -> list[str]:
        """Get fallback list of well-known stocks with good analyst coverage."""
        return [
            # Mega cap tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            # Large cap tech
            "AMD", "NFLX", "CRM", "ORCL", "ADBE",
            # Semiconductors
            "AVGO", "QCOM", "TXN",
            # Finance
            "JPM", "BAC", "GS",
            # Healthcare
            "JNJ", "UNH", "PFE",
            # Consumer
            "WMT", "HD", "MCD",
        ]


class CryptoScreener:
    """Screens and discovers cryptocurrencies dynamically."""

    def __init__(self) -> None:
        """Initialize the crypto screener."""
        self._cache: TTLCache = TTLCache(maxsize=100, ttl=3600)
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

    async def get_top_cryptos(self, limit: int = 20) -> list[str]:
        """Get top cryptocurrencies by market cap.

        Uses CoinGecko API to fetch top cryptos.

        Args:
            limit: Maximum number of cryptos to return

        Returns:
            List of crypto symbols
        """
        cache_key = f"top_cryptos_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        session = await self._get_session()

        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    symbols = [
                        coin.get("symbol", "").upper()
                        for coin in data
                        if coin.get("symbol")
                    ]
                    self._cache[cache_key] = symbols
                    return symbols

        except Exception as e:
            logger.warning(f"Error fetching top cryptos: {e}")

        # Fallback
        fallback = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT",
                    "LINK", "AVAX", "MATIC", "SHIB", "LTC", "UNI", "ATOM"]
        return fallback[:limit]

    async def get_trending_cryptos(self, limit: int = 10) -> list[str]:
        """Get trending cryptocurrencies.

        Args:
            limit: Maximum number of cryptos to return

        Returns:
            List of crypto symbols
        """
        cache_key = f"trending_cryptos_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        session = await self._get_session()

        try:
            url = "https://api.coingecko.com/api/v3/search/trending"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    coins = data.get("coins", [])
                    symbols = [
                        coin.get("item", {}).get("symbol", "").upper()
                        for coin in coins[:limit]
                        if coin.get("item", {}).get("symbol")
                    ]
                    self._cache[cache_key] = symbols
                    return symbols

        except Exception as e:
            logger.warning(f"Error fetching trending cryptos: {e}")

        return []

    async def discover_cryptos(self, limit: int = 20) -> list[str]:
        """Discover cryptos using multiple sources.

        Combines top by market cap and trending.

        Args:
            limit: Maximum total cryptos to return

        Returns:
            List of unique crypto symbols
        """
        cache_key = f"discover_crypto_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        top_task = self.get_top_cryptos(limit)
        trending_task = self.get_trending_cryptos(limit // 2)

        top_cryptos, trending = await asyncio.gather(
            top_task, trending_task,
            return_exceptions=True
        )

        seen = set()
        combined = []

        for source in [top_cryptos, trending]:
            if isinstance(source, Exception):
                continue
            for symbol in source:
                if symbol not in seen:
                    seen.add(symbol)
                    combined.append(symbol)

        result = combined[:limit]
        self._cache[cache_key] = result
        return result
