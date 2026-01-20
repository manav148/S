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
from .data_sources.finviz import FinvizData, FinvizFetcher
from .data_sources.news import NewsAggregator, NewsSentiment
from .data_sources.options import OptionsDataFetcher, OptionsSentiment
from .data_sources.screener import CryptoScreener, StockScreener

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
    options_score: DataSourceScore | None = None
    finviz_score: DataSourceScore | None = None

    # Raw data
    analyst_data: AnalystConsensus | None = None
    betting_data: BettingMarketSentiment | None = None
    news_data: NewsSentiment | None = None
    options_data: OptionsSentiment | None = None
    finviz_data: FinvizData | None = None

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
        if self.options_score:
            breakdown["options"] = self.options_score.score
        if self.finviz_score:
            breakdown["finviz"] = self.finviz_score.score
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

        if self.options_data and self.options_data.confidence != "none":
            if self.options_data.sentiment_score >= 0.6:
                factors.append(
                    f"Options flow: bullish ({self.options_data.signal_summary})"
                )
            if self.options_data.metrics and self.options_data.metrics.put_call_volume_ratio:
                pcr = self.options_data.metrics.put_call_volume_ratio
                if pcr < 0.7:
                    factors.append(f"Put/Call ratio: {pcr:.2f} (bullish)")

        if self.finviz_data:
            # Technical signals
            if self.finviz_data.technicals:
                tech = self.finviz_data.technicals
                if tech.overall_signal.value >= 4:
                    factors.append(f"Finviz technicals: {tech.overall_signal.name}")
                if tech.rsi and tech.rsi <= 35:
                    factors.append(f"RSI oversold: {tech.rsi:.0f}")

            # Valuation signals
            if self.finviz_data.valuation:
                val = self.finviz_data.valuation
                if val.valuation_score >= 0.7:
                    factors.append(f"Finviz valuation: attractive ({val.pe_signal})")
                if val.peg and 0 < val.peg < 1:
                    factors.append(f"PEG ratio: {val.peg:.2f} (undervalued)")

            # Insider activity
            if self.finviz_data.insider_activity and self.finviz_data.insider_activity.trades:
                insider = self.finviz_data.insider_activity
                if insider.sentiment.value >= 4:
                    factors.append(f"Insider activity: {insider.sentiment.name}")

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

        if self.options_data and self.options_data.confidence != "none":
            if self.options_data.sentiment_score < 0.4:
                risks.append(
                    f"Options flow: bearish ({self.options_data.signal_summary})"
                )
            if self.options_data.metrics:
                pcr = self.options_data.metrics.put_call_volume_ratio
                if pcr and pcr > 1.2:
                    risks.append(f"High put/call ratio: {pcr:.2f} (bearish)")
                if self.options_data.metrics.iv_skew and self.options_data.metrics.iv_skew > 0.05:
                    risks.append("Elevated put IV skew (hedging/fear)")

        if self.finviz_data:
            # Technical risks
            if self.finviz_data.technicals:
                tech = self.finviz_data.technicals
                if tech.overall_signal.value <= 2:
                    risks.append(f"Finviz technicals: {tech.overall_signal.name}")
                if tech.rsi and tech.rsi >= 70:
                    risks.append(f"RSI overbought: {tech.rsi:.0f}")

            # Valuation risks
            if self.finviz_data.valuation:
                val = self.finviz_data.valuation
                if val.valuation_score <= 0.3:
                    risks.append(f"Finviz valuation: {val.pe_signal}")
                if val.peg and val.peg > 2:
                    risks.append(f"High PEG ratio: {val.peg:.2f}")

            # Insider selling
            if self.finviz_data.insider_activity and self.finviz_data.insider_activity.trades:
                insider = self.finviz_data.insider_activity
                if insider.sentiment.value <= 2:
                    risks.append(f"Insider activity: {insider.sentiment.name}")

            # Short interest
            if self.finviz_data.short_float and self.finviz_data.short_float > 15:
                risks.append(f"High short interest: {self.finviz_data.short_float:.1f}%")

        return risks

    @property
    def current_price(self) -> float | None:
        """Get current price from analyst data."""
        if self.analyst_data:
            return self.analyst_data.current_price
        return None

    @property
    def target_price(self) -> float | None:
        """Get average analyst target price."""
        if self.analyst_data:
            return self.analyst_data.average_target_price
        return None

    @property
    def upside_potential(self) -> float | None:
        """Calculate potential upside percentage to target price."""
        if self.current_price and self.target_price and self.current_price > 0:
            return ((self.target_price - self.current_price) / self.current_price) * 100
        return None

    @property
    def downside_risk(self) -> float | None:
        """Estimate downside risk percentage.

        Uses ATM implied volatility if available (annualized, scaled to horizon),
        otherwise defaults to 15%.
        """
        # Try to use options IV for a data-driven estimate
        if self.options_data and self.options_data.metrics:
            atm_iv = self.options_data.metrics.atm_iv
            if atm_iv and atm_iv > 0:
                # Scale annualized IV to ~3 month horizon (sqrt of time)
                # ATM IV is already a decimal (e.g., 0.25 = 25%)
                three_month_move = atm_iv * (0.25 ** 0.5)  # sqrt(3/12)
                return three_month_move * 100  # Convert to percentage

        # Default downside estimate based on asset type and volatility assumptions
        if self.asset_type == "crypto":
            return 25.0  # Crypto is more volatile
        return 15.0  # Default for stocks

    @property
    def risk_reward_ratio(self) -> float | None:
        """Calculate risk/reward ratio (upside / downside).

        A ratio > 2 is generally considered favorable.
        Returns None if data is insufficient.
        """
        upside = self.upside_potential
        downside = self.downside_risk

        if upside is not None and downside and downside > 0:
            return upside / downside
        return None

    @property
    def risk_reward_label(self) -> str:
        """Get a label for the risk/reward ratio."""
        rr = self.risk_reward_ratio
        if rr is None:
            return "N/A"
        if rr >= 3.0:
            return "Excellent"
        if rr >= 2.0:
            return "Good"
        if rr >= 1.0:
            return "Fair"
        if rr >= 0.5:
            return "Poor"
        return "Unfavorable"


class RecommendationEngine:
    """Engine that combines all data sources to generate recommendations."""

    def __init__(self) -> None:
        """Initialize the recommendation engine."""
        self.analyst_fetcher = AnalystDataFetcher()
        self.betting_fetcher = BettingMarketFetcher()
        self.news_aggregator = NewsAggregator()
        self.options_fetcher = OptionsDataFetcher()
        self.finviz_fetcher = FinvizFetcher()
        self.stock_screener = StockScreener()
        self.crypto_screener = CryptoScreener()

        # Weights from config
        self._analyst_weight = config.analyst_weight
        self._betting_weight = config.betting_weight
        self._news_weight = config.news_weight
        self._options_weight = config.get("weights", "options", default=0.15)
        self._finviz_weight = config.get("weights", "finviz", default=0.15)

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
            self.options_fetcher.close(),
            self.finviz_fetcher.close(),
            self.stock_screener.close(),
            self.crypto_screener.close(),
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

        # Options and Finviz data only available for stocks
        if asset_type == "stock":
            options_task = self.options_fetcher.get_options_sentiment(symbol)
            finviz_task = self.finviz_fetcher.get_stock_data(symbol)
            analyst_data, betting_data, news_data, options_data, finviz_data = await asyncio.gather(
                analyst_task, betting_task, news_task, options_task, finviz_task
            )
        else:
            analyst_data, betting_data, news_data = await asyncio.gather(
                analyst_task, betting_task, news_task
            )
            options_data = None
            finviz_data = None

        # Calculate individual scores
        analyst_score = self._calculate_analyst_score(analyst_data)
        betting_score = self._calculate_betting_score(betting_data)
        news_score = self._calculate_news_score(news_data)
        options_score = self._calculate_options_score(options_data) if options_data else None
        finviz_score = self._calculate_finviz_score(finviz_data) if finviz_data else None

        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(
            analyst_score, betting_score, news_score, options_score, finviz_score
        )

        # Determine recommendation type
        recommendation_type = self._score_to_recommendation(overall_score)

        # Determine confidence level
        confidence = self._calculate_confidence(
            analyst_score, betting_score, news_score, options_score, finviz_score
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
            options_score=options_score,
            finviz_score=finviz_score,
            analyst_data=analyst_data,
            betting_data=betting_data,
            news_data=news_data,
            options_data=options_data,
            finviz_data=finviz_data,
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

    def _calculate_options_score(
        self, data: OptionsSentiment
    ) -> DataSourceScore:
        """Calculate score from options sentiment data."""
        score = data.sentiment_score
        confidence = data.confidence

        details = {
            "put_call_signal": data.put_call_signal,
            "iv_signal": data.iv_signal,
            "unusual_activity_signal": data.unusual_activity_signal,
            "signal_summary": data.signal_summary,
        }

        if data.metrics:
            details.update({
                "put_call_volume_ratio": data.metrics.put_call_volume_ratio,
                "put_call_oi_ratio": data.metrics.put_call_oi_ratio,
                "iv_skew": data.metrics.iv_skew,
                "atm_iv": data.metrics.atm_iv,
                "total_call_volume": data.metrics.total_call_volume,
                "total_put_volume": data.metrics.total_put_volume,
                "unusual_calls_count": len(data.metrics.unusual_calls),
                "unusual_puts_count": len(data.metrics.unusual_puts),
            })

        return DataSourceScore(
            source="options",
            score=score,
            weight=self._options_weight,
            confidence=confidence,
            details=details,
        )

    def _calculate_finviz_score(
        self, data: FinvizData
    ) -> DataSourceScore:
        """Calculate score from Finviz data."""
        score = data.overall_score
        confidence = "none"

        # Determine confidence based on data completeness
        has_technicals = data.technicals is not None
        has_valuation = data.valuation is not None
        has_insider = data.insider_activity is not None and len(data.insider_activity.trades) > 0

        data_points = sum([has_technicals, has_valuation, has_insider])
        if data_points >= 3:
            confidence = "high"
        elif data_points >= 2:
            confidence = "medium"
        elif data_points >= 1:
            confidence = "low"

        details = {
            "signal_summary": data.signal_summary,
            "price": data.price,
            "change": data.change,
            "short_float": data.short_float,
            "short_ratio": data.short_ratio,
            "analyst_recommendation": data.analyst_recommendation,
            "target_price": data.target_price,
        }

        if data.technicals:
            details.update({
                "rsi": data.technicals.rsi,
                "sma20": data.technicals.sma20,
                "sma50": data.technicals.sma50,
                "sma200": data.technicals.sma200,
                "technical_signal": data.technicals.overall_signal.name,
            })

        if data.valuation:
            details.update({
                "pe": data.valuation.pe,
                "forward_pe": data.valuation.forward_pe,
                "peg": data.valuation.peg,
                "pb": data.valuation.pb,
                "valuation_score": data.valuation.valuation_score,
            })

        if data.insider_activity:
            details.update({
                "insider_buy_count": data.insider_activity.buy_count,
                "insider_sell_count": data.insider_activity.sell_count,
                "insider_sentiment": data.insider_activity.sentiment.name,
            })

        return DataSourceScore(
            source="finviz",
            score=score,
            weight=self._finviz_weight,
            confidence=confidence,
            details=details,
        )

    def _calculate_overall_score(
        self,
        analyst: DataSourceScore,
        betting: DataSourceScore,
        news: DataSourceScore,
        options: DataSourceScore | None = None,
        finviz: DataSourceScore | None = None,
    ) -> float:
        """Calculate weighted overall score.

        Adjusts weights based on data quality/confidence.
        """
        scores = [
            (analyst.score, analyst.weight, analyst.confidence),
            (betting.score, betting.weight, betting.confidence),
            (news.score, news.weight, news.confidence),
        ]

        # Add options if available (stocks only)
        if options is not None:
            scores.append((options.score, options.weight, options.confidence))

        # Add finviz if available (stocks only)
        if finviz is not None:
            scores.append((finviz.score, finviz.weight, finviz.confidence))

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
        options: DataSourceScore | None = None,
        finviz: DataSourceScore | None = None,
    ) -> str:
        """Calculate overall confidence level."""
        confidence_scores = {"high": 3, "medium": 2, "low": 1, "none": 0}

        scores = [
            confidence_scores[analyst.confidence],
            confidence_scores[betting.confidence],
            confidence_scores[news.confidence],
        ]

        # Add options if available (stocks only)
        if options is not None:
            scores.append(confidence_scores[options.confidence])

        # Add finviz if available (stocks only)
        if finviz is not None:
            scores.append(confidence_scores[finviz.confidence])

        avg_score = sum(scores) / len(scores)

        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"

    async def get_top_picks(
        self,
        symbols: list[str] | None = None,
        asset_type: str = "stock",
        top_n: int | None = None,
    ) -> list[Recommendation]:
        """Get top picks from a list of symbols or discover dynamically.

        Args:
            symbols: List of symbols to analyze. If None, discovers dynamically.
            asset_type: 'stock' or 'crypto'
            top_n: Number of top picks to return (defaults to config)

        Returns:
            List of Recommendations sorted by score
        """
        if top_n is None:
            top_n = config.top_n

        # If no symbols provided, discover them dynamically
        if symbols is None:
            symbols = await self.discover_symbols(asset_type, limit=top_n * 2)

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

    async def discover_symbols(
        self, asset_type: str = "stock", limit: int = 30
    ) -> list[str]:
        """Dynamically discover symbols to analyze.

        Uses screeners to find stocks/crypto with high analyst interest,
        trading activity, or market momentum.

        Args:
            asset_type: 'stock' or 'crypto'
            limit: Maximum symbols to discover

        Returns:
            List of symbols to analyze
        """
        if asset_type == "crypto":
            return await self.crypto_screener.discover_cryptos(limit)
        else:
            return await self.stock_screener.discover_stocks(limit)

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


