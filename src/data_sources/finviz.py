"""Finviz data fetcher for technical signals, valuation, and insider trading."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from cachetools import TTLCache

from ..config import config

logger = logging.getLogger(__name__)

# Thread pool for running synchronous finviz calls
_executor = ThreadPoolExecutor(max_workers=5)

# Lazy import finviz to handle if not installed
_finviz = None


def _get_finviz():
    """Lazy import finviz module."""
    global _finviz
    if _finviz is None:
        try:
            import finviz
            _finviz = finviz
        except ImportError:
            logger.warning("finviz package not installed. Run: pip install finviz")
            _finviz = False
    return _finviz if _finviz else None


class TechnicalSignal(Enum):
    """Technical analysis signal strength."""
    STRONG_BULLISH = 5
    BULLISH = 4
    NEUTRAL = 3
    BEARISH = 2
    STRONG_BEARISH = 1


class InsiderSentiment(Enum):
    """Insider trading sentiment."""
    HEAVY_BUYING = 5
    BUYING = 4
    NEUTRAL = 3
    SELLING = 2
    HEAVY_SELLING = 1


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators from Finviz."""
    rsi: float | None = None  # Relative Strength Index (0-100)
    sma20: str | None = None  # Price vs 20-day SMA
    sma50: str | None = None  # Price vs 50-day SMA
    sma200: str | None = None  # Price vs 200-day SMA
    pattern: str | None = None  # Chart pattern detected
    volatility: float | None = None  # Weekly/Monthly volatility
    atr: float | None = None  # Average True Range
    beta: float | None = None  # Stock beta

    @property
    def rsi_signal(self) -> TechnicalSignal:
        """Get signal from RSI."""
        if self.rsi is None:
            return TechnicalSignal.NEUTRAL
        if self.rsi <= 30:
            return TechnicalSignal.STRONG_BULLISH  # Oversold
        if self.rsi <= 40:
            return TechnicalSignal.BULLISH
        if self.rsi >= 70:
            return TechnicalSignal.STRONG_BEARISH  # Overbought
        if self.rsi >= 60:
            return TechnicalSignal.BEARISH
        return TechnicalSignal.NEUTRAL

    @property
    def sma_signal(self) -> TechnicalSignal:
        """Get combined SMA trend signal."""
        signals = []
        for sma in [self.sma20, self.sma50, self.sma200]:
            if sma and "above" in sma.lower():
                signals.append(1)
            elif sma and "below" in sma.lower():
                signals.append(-1)
            else:
                signals.append(0)

        avg = sum(signals) / len(signals) if signals else 0
        if avg > 0.6:
            return TechnicalSignal.STRONG_BULLISH
        if avg > 0.2:
            return TechnicalSignal.BULLISH
        if avg < -0.6:
            return TechnicalSignal.STRONG_BEARISH
        if avg < -0.2:
            return TechnicalSignal.BEARISH
        return TechnicalSignal.NEUTRAL

    @property
    def overall_signal(self) -> TechnicalSignal:
        """Combine all technical signals."""
        rsi_val = self.rsi_signal.value
        sma_val = self.sma_signal.value
        avg = (rsi_val + sma_val) / 2

        if avg >= 4.5:
            return TechnicalSignal.STRONG_BULLISH
        if avg >= 3.5:
            return TechnicalSignal.BULLISH
        if avg <= 1.5:
            return TechnicalSignal.STRONG_BEARISH
        if avg <= 2.5:
            return TechnicalSignal.BEARISH
        return TechnicalSignal.NEUTRAL


@dataclass
class ValuationMetrics:
    """Valuation metrics from Finviz."""
    pe: float | None = None  # Price/Earnings
    forward_pe: float | None = None  # Forward P/E
    peg: float | None = None  # Price/Earnings to Growth
    ps: float | None = None  # Price/Sales
    pb: float | None = None  # Price/Book
    pc: float | None = None  # Price/Cash
    pfcf: float | None = None  # Price/Free Cash Flow

    @property
    def pe_signal(self) -> str:
        """Evaluate P/E ratio."""
        if self.pe is None:
            return "N/A"
        if self.pe < 0:
            return "Negative earnings"
        if self.pe < 15:
            return "Undervalued"
        if self.pe < 25:
            return "Fair value"
        if self.pe < 40:
            return "Growth premium"
        return "Overvalued"

    @property
    def peg_signal(self) -> str:
        """Evaluate PEG ratio (P/E to Growth)."""
        if self.peg is None:
            return "N/A"
        if self.peg < 0:
            return "Negative growth"
        if self.peg < 1:
            return "Undervalued"
        if self.peg < 1.5:
            return "Fair value"
        if self.peg < 2:
            return "Slightly overvalued"
        return "Overvalued"

    @property
    def valuation_score(self) -> float:
        """Calculate overall valuation score (0-1, higher = more attractive)."""
        scores = []

        # P/E scoring
        if self.pe is not None and self.pe > 0:
            if self.pe < 15:
                scores.append(0.9)
            elif self.pe < 20:
                scores.append(0.7)
            elif self.pe < 30:
                scores.append(0.5)
            elif self.pe < 50:
                scores.append(0.3)
            else:
                scores.append(0.1)

        # PEG scoring
        if self.peg is not None and self.peg > 0:
            if self.peg < 1:
                scores.append(0.9)
            elif self.peg < 1.5:
                scores.append(0.7)
            elif self.peg < 2:
                scores.append(0.5)
            else:
                scores.append(0.3)

        # P/B scoring
        if self.pb is not None and self.pb > 0:
            if self.pb < 1:
                scores.append(0.9)
            elif self.pb < 2:
                scores.append(0.7)
            elif self.pb < 4:
                scores.append(0.5)
            else:
                scores.append(0.3)

        return sum(scores) / len(scores) if scores else 0.5


@dataclass
class InsiderTrade:
    """Individual insider trade."""
    insider_name: str
    relationship: str  # CEO, CFO, Director, etc.
    transaction: str  # Buy, Sale, Option Exercise
    shares: int
    value: float | None
    date: datetime | None


@dataclass
class InsiderActivity:
    """Aggregated insider trading activity."""
    trades: list[InsiderTrade] = field(default_factory=list)
    net_shares: int = 0  # Positive = net buying
    buy_count: int = 0
    sell_count: int = 0

    @property
    def sentiment(self) -> InsiderSentiment:
        """Calculate insider sentiment."""
        if not self.trades:
            return InsiderSentiment.NEUTRAL

        ratio = self.buy_count / (self.buy_count + self.sell_count) if (self.buy_count + self.sell_count) > 0 else 0.5

        if ratio > 0.8:
            return InsiderSentiment.HEAVY_BUYING
        if ratio > 0.6:
            return InsiderSentiment.BUYING
        if ratio < 0.2:
            return InsiderSentiment.HEAVY_SELLING
        if ratio < 0.4:
            return InsiderSentiment.SELLING
        return InsiderSentiment.NEUTRAL

    @property
    def sentiment_score(self) -> float:
        """Get sentiment score (0-1)."""
        return (self.sentiment.value - 1) / 4


@dataclass
class FinvizNews:
    """News article from Finviz."""
    timestamp: str
    headline: str
    url: str
    source: str


@dataclass
class FinvizData:
    """Complete Finviz data for a stock."""
    symbol: str
    price: float | None = None
    change: float | None = None  # Daily percent change
    volume: int | None = None
    market_cap: str | None = None

    # Sub-components
    technicals: TechnicalIndicators | None = None
    valuation: ValuationMetrics | None = None
    insider_activity: InsiderActivity | None = None
    news: list[FinvizNews] = field(default_factory=list)

    # Finviz analyst data
    target_price: float | None = None
    analyst_recommendation: str | None = None  # Buy, Hold, Sell, etc.

    # Additional metrics
    short_float: float | None = None  # Short interest as % of float
    short_ratio: float | None = None  # Days to cover
    earnings_date: str | None = None

    @property
    def overall_score(self) -> float:
        """Calculate overall Finviz score (0-1)."""
        scores = []
        weights = []

        # Technical score (30% weight)
        if self.technicals:
            tech_score = (self.technicals.overall_signal.value - 1) / 4
            scores.append(tech_score)
            weights.append(0.30)

        # Valuation score (30% weight)
        if self.valuation:
            scores.append(self.valuation.valuation_score)
            weights.append(0.30)

        # Insider sentiment (25% weight)
        if self.insider_activity:
            scores.append(self.insider_activity.sentiment_score)
            weights.append(0.25)

        # Short interest inverse (15% weight) - lower short interest is better
        if self.short_float is not None:
            short_score = max(0, 1 - (self.short_float / 30))  # 30% short = 0 score
            scores.append(short_score)
            weights.append(0.15)

        if not scores:
            return 0.5

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    @property
    def signal_summary(self) -> str:
        """Get a summary of the Finviz signals."""
        signals = []

        if self.technicals:
            signals.append(f"Tech: {self.technicals.overall_signal.name}")

        if self.valuation:
            signals.append(f"Val: {self.valuation.pe_signal}")

        if self.insider_activity and self.insider_activity.trades:
            signals.append(f"Insider: {self.insider_activity.sentiment.name}")

        if self.short_float:
            signals.append(f"Short: {self.short_float:.1f}%")

        return " | ".join(signals) if signals else "No data"


class FinvizFetcher:
    """Fetches data from Finviz."""

    def __init__(self) -> None:
        """Initialize the Finviz data fetcher."""
        self._cache: TTLCache = TTLCache(
            maxsize=500,
            ttl=config.get("cache", "ttl_minutes", default=30) * 60,
        )

    async def get_stock_data(self, symbol: str) -> FinvizData | None:
        """Fetch comprehensive Finviz data for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            FinvizData with all available metrics, or None if unavailable
        """
        cache_key = f"finviz_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        finviz = _get_finviz()
        if not finviz:
            return None

        # Run synchronous finviz calls in thread pool
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            _executor, self._fetch_finviz_data, symbol
        )

        if data:
            self._cache[cache_key] = data
        return data

    def _fetch_finviz_data(self, symbol: str) -> FinvizData | None:
        """Fetch all Finviz data synchronously (runs in executor)."""
        finviz = _get_finviz()
        if not finviz:
            return None

        try:
            # Get stock overview (90+ data points)
            stock = finviz.get_stock(symbol)

            # Parse the data
            data = FinvizData(
                symbol=symbol,
                price=self._parse_float(stock.get("Price")),
                change=self._parse_percent(stock.get("Change")),
                volume=self._parse_int(stock.get("Volume")),
                market_cap=stock.get("Market Cap"),
                target_price=self._parse_float(stock.get("Target Price")),
                analyst_recommendation=stock.get("Recom"),
                short_float=self._parse_percent(stock.get("Short Float")),
                short_ratio=self._parse_float(stock.get("Short Ratio")),
                earnings_date=stock.get("Earnings"),
            )

            # Parse technical indicators
            data.technicals = TechnicalIndicators(
                rsi=self._parse_float(stock.get("RSI (14)")),
                sma20=stock.get("SMA20"),
                sma50=stock.get("SMA50"),
                sma200=stock.get("SMA200"),
                volatility=self._parse_percent(stock.get("Volatility W")) or self._parse_percent(stock.get("Volatility M")),
                atr=self._parse_float(stock.get("ATR")),
                beta=self._parse_float(stock.get("Beta")),
            )

            # Parse valuation metrics
            data.valuation = ValuationMetrics(
                pe=self._parse_float(stock.get("P/E")),
                forward_pe=self._parse_float(stock.get("Forward P/E")),
                peg=self._parse_float(stock.get("PEG")),
                ps=self._parse_float(stock.get("P/S")),
                pb=self._parse_float(stock.get("P/B")),
                pc=self._parse_float(stock.get("P/C")),
                pfcf=self._parse_float(stock.get("P/FCF")),
            )

            # Fetch insider trading
            try:
                insiders = finviz.get_insider(symbol)
                data.insider_activity = self._parse_insider_data(insiders)
            except Exception as e:
                logger.debug(f"Could not get insider data for {symbol}: {e}")
                data.insider_activity = InsiderActivity()

            # Fetch news
            try:
                news = finviz.get_news(symbol)
                data.news = self._parse_news(news)
            except Exception as e:
                logger.debug(f"Could not get news for {symbol}: {e}")

            return data

        except Exception as e:
            logger.error(f"Error fetching Finviz data for {symbol}: {e}")
            return None

    def _parse_float(self, value: Any) -> float | None:
        """Parse a float value from Finviz data."""
        if value is None or value == "-" or value == "":
            return None
        try:
            # Handle values like "1.5B", "150M", "15K"
            if isinstance(value, str):
                value = value.replace(",", "").strip()
                multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
                for suffix, mult in multipliers.items():
                    if value.endswith(suffix):
                        return float(value[:-1]) * mult
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value: Any) -> int | None:
        """Parse an integer value."""
        parsed = self._parse_float(value)
        return int(parsed) if parsed is not None else None

    def _parse_percent(self, value: Any) -> float | None:
        """Parse a percentage value (returns as decimal, e.g., 5.5 for 5.5%)."""
        if value is None or value == "-" or value == "":
            return None
        try:
            if isinstance(value, str):
                value = value.replace("%", "").replace(",", "").strip()
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_insider_data(self, insiders: list) -> InsiderActivity:
        """Parse insider trading data."""
        trades = []
        buy_count = 0
        sell_count = 0
        net_shares = 0

        for trade in insiders[:20]:  # Last 20 trades
            try:
                transaction = trade.get("Transaction", "").lower()
                shares = self._parse_int(trade.get("Shares")) or 0

                is_buy = "buy" in transaction or "purchase" in transaction
                is_sell = "sale" in transaction or "sell" in transaction

                if is_buy:
                    buy_count += 1
                    net_shares += shares
                elif is_sell:
                    sell_count += 1
                    net_shares -= shares

                trades.append(InsiderTrade(
                    insider_name=trade.get("Insider Trading", "Unknown"),
                    relationship=trade.get("Relationship", "Unknown"),
                    transaction=trade.get("Transaction", "Unknown"),
                    shares=shares,
                    value=self._parse_float(trade.get("Value")),
                    date=None,  # Would need to parse date string
                ))
            except Exception as e:
                logger.debug(f"Error parsing insider trade: {e}")
                continue

        return InsiderActivity(
            trades=trades,
            net_shares=net_shares,
            buy_count=buy_count,
            sell_count=sell_count,
        )

    def _parse_news(self, news: list) -> list[FinvizNews]:
        """Parse news articles."""
        articles = []
        for item in news[:10]:  # Last 10 articles
            try:
                # Finviz returns tuples: (timestamp, headline, url, source)
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    articles.append(FinvizNews(
                        timestamp=str(item[0]),
                        headline=str(item[1]),
                        url=str(item[2]),
                        source=str(item[3]),
                    ))
            except Exception as e:
                logger.debug(f"Error parsing news item: {e}")
                continue
        return articles

    async def get_multiple_stocks(self, symbols: list[str]) -> dict[str, FinvizData | None]:
        """Fetch Finviz data for multiple stocks.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dict mapping symbol to FinvizData
        """
        tasks = [self.get_stock_data(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result if isinstance(result, FinvizData) else None
            for symbol, result in zip(symbols, results)
        }

    async def close(self) -> None:
        """Close any resources (placeholder for consistency)."""
        pass
