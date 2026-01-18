"""Recommendation engine that combines all data sources."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .config import config
from .data_sources.analyst import AnalystConsensus, AnalystDataFetcher
from .data_sources.betting import BettingMarketFetcher, BettingMarketSentiment
from .data_sources.news import NewsAggregator, NewsSentiment

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Recommendation types."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


@dataclass
class DataSourceScore:
    """Score from a single data source."""

    source: str
    score: float  # 0-1 normalized score
    weight: float
    confidence: str  # 'high', 'medium', 'low', 'none'
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Complete recommendation for an asset."""

    symbol: str
    asset_type: str
    recommendation: RecommendationType
    overall_score: float  # 0-1 weighted average
    confidence: str  # 'high', 'medium', 'low'

    # Individual data source scores
    analyst_score: DataSourceScore | None = None
    betting_score: DataSourceScore | None = None
    news_score: DataSourceScore | None = None

    # Raw data
    analyst_data: AnalystConsensus | None = None
    betting_data: BettingMarketSentiment | None = None
    news_data: NewsSentiment | None = None

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    horizon_months: tuple[int, int] = (6, 12)

    @property
    def score_breakdown(self) -> dict[str, float]:
        """Get breakdown of scores by source."""
        breakdown = {}
        if self.analyst_score:
            breakdown["analyst"] = self.analyst_score.score
        if self.betting_score:
            breakdown["betting"] = self.betting_score.score
        if self.news_score:
            breakdown["news"] = self.news_score.score
        return breakdown

    @property
    def supporting_factors(self) -> list[str]:
        """Get list of supporting factors for the recommendation."""
        factors = []

        if self.analyst_data:
            if self.analyst_data.consensus_score >= 0.6:
                factors.append(
                    f"Analyst consensus: {self.analyst_data.consensus_rating} "
                    f"({self.analyst_data.total_analysts} analysts)"
                )
            if self.analyst_data.average_upside:
                factors.append(
                    f"Average target upside: {self.analyst_data.average_upside:.1f}%"
                )

        if self.betting_data and self.betting_data.markets:
            factors.append(
                f"Prediction markets: {self.betting_data.sentiment_score:.0%} bullish "
                f"(${self.betting_data.total_volume:,.0f} volume)"
            )

        if self.news_data and self.news_data.articles:
            factors.append(
                f"News sentiment: {self.news_data.sentiment_label} "
                f"({self.news_data.article_count} articles)"
            )

        return factors

    @property
    def risk_factors(self) -> list[str]:
        """Get list of risk factors or concerns."""
        risks = []

        if self.analyst_data:
            if self.analyst_data.total_analysts < config.analyst_min_count:
                risks.append(
                    f"Limited analyst coverage ({self.analyst_data.total_analysts} analysts)"
                )

        if self.betting_data:
            if self.betting_data.confidence_level in ("low", "very_low"):
                risks.append("Low prediction market liquidity")

        if self.news_data:
            if self.news_data.bearish_count > self.news_data.bullish_count:
                risks.append(
                    f"Negative news sentiment ({self.news_data.bearish_count} bearish articles)"
                )

        return risks


class RecommendationEngine:
    """Engine that combines all data sources to generate recommendations."""

    def __init__(self) -> None:
        """Initialize the recommendation engine."""
        self.analyst_fetcher = AnalystDataFetcher()
        self.betting_fetcher = BettingMarketFetcher()
        self.news_aggregator = NewsAggregator()

        # Weights from config
        self._analyst_weight = config.analyst_weight
        self._betting_weight = config.betting_weight
        self._news_weight = config.news_weight

        # Thresholds
        self._strong_buy_threshold = config.get(
            "recommendation", "strong_buy_threshold", default=0.8
        )
        self._buy_threshold = config.get(
            "recommendation", "buy_threshold", default=0.6
        )
        self._hold_threshold = config.get(
            "recommendation", "hold_threshold", default=0.4
        )

    async def close(self) -> None:
        """Close all data source connections."""
        await asyncio.gather(
            self.analyst_fetcher.close(),
            self.betting_fetcher.close(),
            self.news_aggregator.close(),
        )

    async def get_recommendation(
        self, symbol: str, asset_type: str = "stock"
    ) -> Recommendation:
        """Get a complete recommendation for an asset.

        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTC')
            asset_type: 'stock' or 'crypto'

        Returns:
            Complete Recommendation with all data sources
        """
        # Fetch data from all sources concurrently
        analyst_task = (
            self.analyst_fetcher.get_stock_analysts(symbol)
            if asset_type == "stock"
            else self.analyst_fetcher.get_crypto_analysts(symbol)
        )
        betting_task = self.betting_fetcher.get_market_sentiment(symbol, asset_type)
        news_task = self.news_aggregator.get_news_sentiment(symbol, asset_type)

        analyst_data, betting_data, news_data = await asyncio.gather(
            analyst_task, betting_task, news_task
        )

        # Calculate individual scores
        analyst_score = self._calculate_analyst_score(analyst_data)
        betting_score = self._calculate_betting_score(betting_data)
        news_score = self._calculate_news_score(news_data)

        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(
            analyst_score, betting_score, news_score
        )

        # Determine recommendation type
        recommendation_type = self._score_to_recommendation(overall_score)

        # Determine confidence level
        confidence = self._calculate_confidence(
            analyst_score, betting_score, news_score
        )

        return Recommendation(
            symbol=symbol,
            asset_type=asset_type,
            recommendation=recommendation_type,
            overall_score=overall_score,
            confidence=confidence,
            analyst_score=analyst_score,
            betting_score=betting_score,
            news_score=news_score,
            analyst_data=analyst_data,
            betting_data=betting_data,
            news_data=news_data,
            horizon_months=config.horizon_months,
        )

    def _calculate_analyst_score(
        self, data: AnalystConsensus
    ) -> DataSourceScore:
        """Calculate score from analyst data."""
        score = data.consensus_score if data.recommendations else 0.5
        confidence = "none"

        if data.recommendations:
            if data.total_analysts >= config.analyst_min_count:
                confidence = "high"
            elif data.total_analysts >= config.analyst_min_count // 2:
                confidence = "medium"
            else:
                confidence = "low"

        details = {
            "total_analysts": data.total_analysts,
            "consensus_rating": data.consensus_rating,
            "rating_distribution": data.rating_distribution,
            "average_target": data.average_target_price,
            "average_upside": data.average_upside,
        }

        return DataSourceScore(
            source="analyst",
            score=score,
            weight=self._analyst_weight,
            confidence=confidence,
            details=details,
        )

    def _calculate_betting_score(
        self, data: BettingMarketSentiment
    ) -> DataSourceScore:
        """Calculate score from betting market data."""
        score = data.sentiment_score if data.markets else 0.5
        confidence = data.confidence_level if data.markets else "none"

        details = {
            "market_count": data.market_count,
            "total_volume": data.total_volume,
            "average_probability": data.average_bullish_probability,
        }

        return DataSourceScore(
            source="betting",
            score=score,
            weight=self._betting_weight,
            confidence=confidence,
            details=details,
        )

    def _calculate_news_score(self, data: NewsSentiment) -> DataSourceScore:
        """Calculate score from news sentiment."""
        score = data.sentiment_score if data.articles else 0.5

        confidence = "none"
        if data.article_count > 0:
            if data.article_count >= 10:
                confidence = "high"
            elif data.article_count >= 5:
                confidence = "medium"
            else:
                confidence = "low"

        details = {
            "article_count": data.article_count,
            "bullish_count": data.bullish_count,
            "bearish_count": data.bearish_count,
            "neutral_count": data.neutral_count,
            "sentiment_label": data.sentiment_label,
        }

        return DataSourceScore(
            source="news",
            score=score,
            weight=self._news_weight,
            confidence=confidence,
            details=details,
        )

    def _calculate_overall_score(
        self,
        analyst: DataSourceScore,
        betting: DataSourceScore,
        news: DataSourceScore,
    ) -> float:
        """Calculate weighted overall score.

        Adjusts weights based on data quality/confidence.
        """
        scores = [
            (analyst.score, analyst.weight, analyst.confidence),
            (betting.score, betting.weight, betting.confidence),
            (news.score, news.weight, news.confidence),
        ]

        # Confidence multipliers
        confidence_mult = {"high": 1.0, "medium": 0.7, "low": 0.4, "none": 0.1}

        weighted_sum = 0.0
        weight_sum = 0.0

        for score, weight, confidence in scores:
            adj_weight = weight * confidence_mult[confidence]
            weighted_sum += score * adj_weight
            weight_sum += adj_weight

        if weight_sum == 0:
            return 0.5  # Neutral if no data

        return weighted_sum / weight_sum

    def _score_to_recommendation(self, score: float) -> RecommendationType:
        """Convert score to recommendation type."""
        if score >= self._strong_buy_threshold:
            return RecommendationType.STRONG_BUY
        elif score >= self._buy_threshold:
            return RecommendationType.BUY
        elif score >= self._hold_threshold:
            return RecommendationType.HOLD
        elif score >= 1 - self._buy_threshold:
            return RecommendationType.SELL
        else:
            return RecommendationType.STRONG_SELL

    def _calculate_confidence(
        self,
        analyst: DataSourceScore,
        betting: DataSourceScore,
        news: DataSourceScore,
    ) -> str:
        """Calculate overall confidence level."""
        confidence_scores = {"high": 3, "medium": 2, "low": 1, "none": 0}

        scores = [
            confidence_scores[analyst.confidence],
            confidence_scores[betting.confidence],
            confidence_scores[news.confidence],
        ]

        avg_score = sum(scores) / len(scores)

        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"

    async def get_top_picks(
        self,
        symbols: list[str],
        asset_type: str = "stock",
        top_n: int | None = None,
    ) -> list[Recommendation]:
        """Get top picks from a list of symbols.

        Args:
            symbols: List of symbols to analyze
            asset_type: 'stock' or 'crypto'
            top_n: Number of top picks to return (defaults to config)

        Returns:
            List of Recommendations sorted by score
        """
        if top_n is None:
            top_n = config.top_n

        # Get recommendations for all symbols
        tasks = [self.get_recommendation(s, asset_type) for s in symbols]
        recommendations = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_recs = [
            r for r in recommendations if isinstance(r, Recommendation)
        ]

        # Sort by overall score descending
        valid_recs.sort(key=lambda x: x.overall_score, reverse=True)

        return valid_recs[:top_n]

    async def get_market_overview(self) -> dict[str, Any]:
        """Get overall market sentiment and overview.

        Returns:
            Dictionary with market overview data
        """
        # Get macro sentiment from prediction markets
        macro_sentiment = await self.betting_fetcher.get_general_market_sentiment()

        # Get trending topics from news
        trending = await self.news_aggregator.get_trending_topics()

        # Get latest headlines
        headlines = await self.news_aggregator.get_latest_headlines(5)

        return {
            "macro_sentiment": macro_sentiment,
            "trending_topics": trending[:10],
            "latest_headlines": [
                {"title": h.title, "source": h.source, "url": h.url}
                for h in headlines
            ],
            "generated_at": datetime.now().isoformat(),
        }


# Default symbol lists for quick analysis
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "NFLX", "CRM",
]

DEFAULT_CRYPTO = [
    "BTC", "ETH", "SOL", "XRP", "ADA",
    "DOGE", "DOT", "LINK", "AVAX", "MATIC",
]
