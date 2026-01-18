"""News aggregation service for market sentiment analysis."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dateutil import parser as date_parser

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Individual news article."""

    title: str
    summary: str
    url: str
    source: str
    published_date: datetime
    sentiment_score: float | None = None  # -1 to 1
    related_symbols: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class NewsSentiment:
    """Aggregated news sentiment for an asset."""

    symbol: str
    asset_type: str
    articles: list[NewsArticle] = field(default_factory=list)
    average_sentiment: float = 0.0
    article_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    @property
    def sentiment_score(self) -> float:
        """Calculate overall sentiment score (0-1).

        Maps from -1 to 1 scale to 0 to 1 scale for consistency.
        """
        if not self.articles:
            return 0.5  # Neutral if no data

        # Average the sentiment scores
        scores = [a.sentiment_score for a in self.articles if a.sentiment_score is not None]
        if not scores:
            return 0.5

        avg = sum(scores) / len(scores)
        # Convert from [-1, 1] to [0, 1]
        return (avg + 1) / 2

    @property
    def sentiment_label(self) -> str:
        """Get human-readable sentiment label."""
        score = self.sentiment_score
        if score >= 0.7:
            return "Very Bullish"
        elif score >= 0.55:
            return "Bullish"
        elif score >= 0.45:
            return "Neutral"
        elif score >= 0.3:
            return "Bearish"
        else:
            return "Very Bearish"


class NewsAggregator:
    """Aggregates and analyzes news from multiple sources."""

    def __init__(self) -> None:
        """Initialize the news aggregator."""
        self._cache: TTLCache = TTLCache(
            maxsize=500,
            ttl=config.get("cache", "ttl_minutes", default=30) * 60,
        )
        self._session: aiohttp.ClientSession | None = None
        self._rss_feeds = config.get("news", "rss_feeds", default=[])
        self._max_age_days = config.get("news", "max_age_days", default=7)
        self._max_articles = config.get("news", "max_articles_per_source", default=20)

        # Sentiment keywords for basic analysis
        self._bullish_words = {
            "surge", "soar", "rally", "bullish", "growth", "gains", "profit",
            "record", "high", "breakthrough", "success", "strong", "beat",
            "exceed", "upgrade", "buy", "outperform", "positive", "boom",
            "momentum", "optimistic", "recovery", "rebound", "jump", "spike",
        }
        self._bearish_words = {
            "crash", "plunge", "drop", "bearish", "decline", "loss", "fall",
            "low", "warning", "concern", "weak", "miss", "fail", "downgrade",
            "sell", "underperform", "negative", "slump", "recession", "fear",
            "pessimistic", "downturn", "sink", "tumble", "collapse",
        }

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

    async def get_news_sentiment(
        self, symbol: str, asset_type: str = "stock"
    ) -> NewsSentiment:
        """Get news sentiment for an asset.

        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTC')
            asset_type: 'stock' or 'crypto'

        Returns:
            NewsSentiment with aggregated news data
        """
        cache_key = f"news_{asset_type}_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch from multiple sources
        articles = await self._fetch_all_news(symbol, asset_type)

        # Filter and score articles
        scored_articles = []
        bullish = bearish = neutral = 0

        for article in articles:
            if self._is_relevant(article, symbol, asset_type):
                article.sentiment_score = self._analyze_sentiment(article)
                scored_articles.append(article)

                if article.sentiment_score > 0.1:
                    bullish += 1
                elif article.sentiment_score < -0.1:
                    bearish += 1
                else:
                    neutral += 1

        sentiment = NewsSentiment(
            symbol=symbol,
            asset_type=asset_type,
            articles=scored_articles,
            average_sentiment=(
                sum(a.sentiment_score for a in scored_articles) / len(scored_articles)
                if scored_articles
                else 0.0
            ),
            article_count=len(scored_articles),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )

        self._cache[cache_key] = sentiment
        return sentiment

    async def _fetch_all_news(
        self, symbol: str, asset_type: str
    ) -> list[NewsArticle]:
        """Fetch news from all configured sources."""
        articles: list[NewsArticle] = []

        # Fetch from RSS feeds
        tasks = []
        for feed_config in self._rss_feeds:
            tasks.append(
                self._fetch_rss_feed(
                    feed_config.get("url", ""),
                    feed_config.get("name", "Unknown"),
                    symbol,
                )
            )

        # Add direct API fetches for specific sources
        tasks.append(self._fetch_google_news(symbol, asset_type))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"News fetch error: {result}")

        return articles

    async def _fetch_rss_feed(
        self, url: str, source_name: str, symbol: str
    ) -> list[NewsArticle]:
        """Fetch and parse an RSS feed."""
        if not url:
            return []

        articles: list[NewsArticle] = []
        session = await self._get_session()

        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    cutoff_date = datetime.now(timezone.utc) - timedelta(
                        days=self._max_age_days
                    )

                    for entry in feed.entries[: self._max_articles]:
                        pub_date = self._parse_date(entry.get("published"))
                        if pub_date and pub_date < cutoff_date:
                            continue

                        title = entry.get("title", "")
                        summary = self._clean_html(
                            entry.get("summary", entry.get("description", ""))
                        )

                        articles.append(
                            NewsArticle(
                                title=title,
                                summary=summary[:1000],
                                url=entry.get("link", ""),
                                source=source_name,
                                published_date=pub_date or datetime.now(timezone.utc),
                                related_symbols=[symbol],
                            )
                        )

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching RSS feed: {url}")
        except Exception as e:
            logger.debug(f"Error fetching RSS feed {url}: {e}")

        return articles

    async def _fetch_google_news(
        self, symbol: str, asset_type: str
    ) -> list[NewsArticle]:
        """Fetch news from Google News RSS."""
        articles: list[NewsArticle] = []

        # Build search query
        search_terms = self._get_search_terms(symbol, asset_type)
        query = "+".join(search_terms[:2])  # Use first two terms

        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        session = await self._get_session()

        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    cutoff_date = datetime.now(timezone.utc) - timedelta(
                        days=self._max_age_days
                    )

                    for entry in feed.entries[: self._max_articles]:
                        pub_date = self._parse_date(entry.get("published"))
                        if pub_date and pub_date < cutoff_date:
                            continue

                        title = entry.get("title", "")
                        summary = self._clean_html(
                            entry.get("summary", entry.get("description", ""))
                        )

                        # Extract source from title (Google News format: "Title - Source")
                        source = "Google News"
                        if " - " in title:
                            parts = title.rsplit(" - ", 1)
                            if len(parts) == 2:
                                title, source = parts

                        articles.append(
                            NewsArticle(
                                title=title,
                                summary=summary[:1000],
                                url=entry.get("link", ""),
                                source=source,
                                published_date=pub_date or datetime.now(timezone.utc),
                                related_symbols=[symbol],
                            )
                        )

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching Google News for {symbol}")
        except Exception as e:
            logger.debug(f"Error fetching Google News: {e}")

        return articles

    def _get_search_terms(self, symbol: str, asset_type: str) -> list[str]:
        """Get search terms for a symbol."""
        terms = [symbol]

        stock_names = {
            "AAPL": "Apple",
            "GOOGL": "Google",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta",
            "NVDA": "NVIDIA",
            "AMD": "AMD",
        }

        crypto_names = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "Solana",
            "XRP": "Ripple",
            "DOGE": "Dogecoin",
            "ADA": "Cardano",
        }

        name_map = crypto_names if asset_type == "crypto" else stock_names
        if symbol.upper() in name_map:
            terms.append(name_map[symbol.upper()])

        if asset_type == "crypto":
            terms.append("cryptocurrency")
        else:
            terms.append("stock")

        return terms

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse a date string into datetime."""
        if not date_str:
            return None
        try:
            parsed = date_parser.parse(date_str)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, TypeError):
            return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    def _is_relevant(
        self, article: NewsArticle, symbol: str, asset_type: str
    ) -> bool:
        """Check if an article is relevant to the given symbol."""
        search_terms = self._get_search_terms(symbol, asset_type)
        combined_text = f"{article.title} {article.summary}".lower()

        return any(term.lower() in combined_text for term in search_terms)

    def _analyze_sentiment(self, article: NewsArticle) -> float:
        """Analyze sentiment of an article.

        Uses keyword-based sentiment analysis.
        Returns score from -1 (very bearish) to 1 (very bullish).
        """
        text = f"{article.title} {article.summary}".lower()
        words = set(re.findall(r"\b\w+\b", text))

        bullish_count = len(words & self._bullish_words)
        bearish_count = len(words & self._bearish_words)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        # Normalize to -1 to 1
        return (bullish_count - bearish_count) / total

    async def get_trending_topics(self) -> list[dict[str, Any]]:
        """Get trending financial topics from news sources.

        Returns list of trending topics with their frequency.
        """
        session = await self._get_session()
        topic_counts: dict[str, int] = {}

        # Fetch from general financial news feeds
        for feed_config in self._rss_feeds:
            try:
                url = feed_config.get("url", "")
                if not url:
                    continue

                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)

                        for entry in feed.entries[:20]:
                            title = entry.get("title", "").lower()

                            # Extract key terms
                            for term in self._extract_key_terms(title):
                                topic_counts[term] = topic_counts.get(term, 0) + 1

            except Exception as e:
                logger.debug(f"Error fetching trending topics: {e}")

        # Sort by count and return top topics
        sorted_topics = sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )

        return [{"topic": t, "count": c} for t, c in sorted_topics[:20]]

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key financial terms from text."""
        # Common stock/crypto symbols
        symbols = re.findall(r"\b[A-Z]{2,5}\b", text.upper())

        # Filter to likely symbols
        valid_symbols = [s for s in symbols if len(s) >= 2 and s.isalpha()]

        return valid_symbols

    async def get_latest_headlines(
        self, count: int = 10
    ) -> list[NewsArticle]:
        """Get latest financial headlines from all sources.

        Args:
            count: Number of headlines to return

        Returns:
            List of most recent NewsArticle objects
        """
        articles = await self._fetch_all_news("", "general")

        # Sort by date, most recent first
        articles.sort(key=lambda x: x.published_date, reverse=True)

        return articles[:count]
