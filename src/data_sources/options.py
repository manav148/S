"""Options market data fetcher for sentiment analysis."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import yfinance as yf
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=5)


@dataclass
class OptionsMetrics:
    """Calculated options metrics for sentiment analysis."""

    symbol: str

    # Put/Call Ratios
    put_call_volume_ratio: float | None = None  # < 0.7 bullish, > 1.0 bearish
    put_call_oi_ratio: float | None = None  # Open interest based

    # Volatility Metrics
    avg_iv_calls: float | None = None  # Average IV for calls
    avg_iv_puts: float | None = None  # Average IV for puts
    iv_skew: float | None = None  # Put IV - Call IV (positive = fear)
    atm_iv: float | None = None  # At-the-money implied volatility

    # Volume Analysis
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0

    # Unusual Activity
    unusual_calls: list[dict] = field(default_factory=list)
    unusual_puts: list[dict] = field(default_factory=list)
    unusual_activity_score: float = 0.0  # 0-1, higher = more unusual activity

    # Price Context
    current_price: float | None = None
    expiration_date: str | None = None

    # Data Quality
    has_data: bool = False
    contracts_analyzed: int = 0


@dataclass
class OptionsSentiment:
    """Aggregated options sentiment for scoring."""

    symbol: str
    sentiment_score: float  # 0-1, higher = more bullish
    confidence: str  # 'high', 'medium', 'low', 'none'
    metrics: OptionsMetrics | None = None

    # Breakdown
    put_call_signal: str = "neutral"  # 'bullish', 'neutral', 'bearish'
    iv_signal: str = "neutral"  # 'low_iv', 'normal', 'high_iv'
    unusual_activity_signal: str = "none"  # 'bullish_flow', 'bearish_flow', 'none'

    @property
    def signal_summary(self) -> str:
        """Get a summary of the options signals."""
        signals = []
        if self.put_call_signal != "neutral":
            signals.append(f"P/C: {self.put_call_signal}")
        if self.iv_signal != "normal":
            signals.append(f"IV: {self.iv_signal}")
        if self.unusual_activity_signal != "none":
            signals.append(f"Flow: {self.unusual_activity_signal}")
        return ", ".join(signals) if signals else "neutral"


class OptionsDataFetcher:
    """Fetches and analyzes options data for sentiment signals."""

    def __init__(self) -> None:
        """Initialize the options data fetcher."""
        self._cache: TTLCache = TTLCache(maxsize=200, ttl=900)  # 15 min cache

    async def close(self) -> None:
        """Close resources (no persistent connections needed)."""
        pass

    async def get_options_sentiment(
        self, symbol: str, days_out: int = 30
    ) -> OptionsSentiment:
        """Get options-based sentiment for a stock.

        Args:
            symbol: Stock symbol
            days_out: Days to expiration to analyze (default 30)

        Returns:
            OptionsSentiment with calculated scores
        """
        cache_key = f"options_{symbol}_{days_out}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch options data in thread pool (yfinance is sync)
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            _executor, self._fetch_options_metrics, symbol, days_out
        )

        # Calculate sentiment from metrics
        sentiment = self._calculate_sentiment(symbol, metrics)

        self._cache[cache_key] = sentiment
        return sentiment

    def _fetch_options_metrics(self, symbol: str, days_out: int) -> OptionsMetrics:
        """Synchronously fetch and calculate options metrics."""
        metrics = OptionsMetrics(symbol=symbol)

        try:
            ticker = yf.Ticker(symbol)

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                logger.debug(f"No options available for {symbol}")
                return metrics

            # Find expiration closest to target days out
            target_date = datetime.now() + timedelta(days=days_out)
            best_exp = self._find_best_expiration(expirations, target_date)

            if not best_exp:
                return metrics

            metrics.expiration_date = best_exp

            # Get options chain
            chain = ticker.option_chain(best_exp)
            calls_df = chain.calls
            puts_df = chain.puts

            # Get current price from underlying
            underlying = chain.underlying
            if underlying and 'regularMarketPrice' in underlying:
                metrics.current_price = underlying['regularMarketPrice']
            else:
                # Fallback to ticker info
                info = ticker.info
                metrics.current_price = info.get('regularMarketPrice') or info.get('currentPrice')

            if metrics.current_price is None:
                logger.debug(f"Could not get current price for {symbol}")
                return metrics

            # Calculate metrics
            metrics = self._calculate_metrics(metrics, calls_df, puts_df)
            metrics.has_data = True

        except Exception as e:
            logger.warning(f"Error fetching options for {symbol}: {e}")

        return metrics

    def _find_best_expiration(
        self, expirations: tuple, target_date: datetime
    ) -> str | None:
        """Find the expiration date closest to target."""
        if not expirations:
            return None

        best_exp = None
        min_diff = float('inf')

        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                diff = abs((exp_date - target_date).days)
                if diff < min_diff:
                    min_diff = diff
                    best_exp = exp
            except ValueError:
                continue

        return best_exp

    def _calculate_metrics(
        self, metrics: OptionsMetrics, calls_df, puts_df
    ) -> OptionsMetrics:
        """Calculate all options metrics from the chain data."""
        import pandas as pd

        current_price = metrics.current_price
        if current_price is None:
            return metrics

        # Filter to liquid contracts (has volume or OI)
        calls = calls_df[
            (calls_df['volume'].notna()) | (calls_df['openInterest'] > 0)
        ].copy()
        puts = puts_df[
            (puts_df['volume'].notna()) | (puts_df['openInterest'] > 0)
        ].copy()

        # Fill NaN volumes with 0
        calls['volume'] = calls['volume'].fillna(0)
        puts['volume'] = puts['volume'].fillna(0)

        metrics.contracts_analyzed = len(calls) + len(puts)

        # === Volume and OI totals ===
        metrics.total_call_volume = int(calls['volume'].sum())
        metrics.total_put_volume = int(puts['volume'].sum())
        metrics.total_call_oi = int(calls['openInterest'].sum())
        metrics.total_put_oi = int(puts['openInterest'].sum())

        # === Put/Call Ratios ===
        if metrics.total_call_volume > 0:
            metrics.put_call_volume_ratio = (
                metrics.total_put_volume / metrics.total_call_volume
            )

        if metrics.total_call_oi > 0:
            metrics.put_call_oi_ratio = (
                metrics.total_put_oi / metrics.total_call_oi
            )

        # === Implied Volatility Analysis ===
        # Weight IV by open interest for more accurate measure
        if len(calls) > 0 and calls['openInterest'].sum() > 0:
            calls_with_iv = calls[calls['impliedVolatility'].notna()]
            if len(calls_with_iv) > 0:
                weights = calls_with_iv['openInterest'] / calls_with_iv['openInterest'].sum()
                metrics.avg_iv_calls = (calls_with_iv['impliedVolatility'] * weights).sum()

        if len(puts) > 0 and puts['openInterest'].sum() > 0:
            puts_with_iv = puts[puts['impliedVolatility'].notna()]
            if len(puts_with_iv) > 0:
                weights = puts_with_iv['openInterest'] / puts_with_iv['openInterest'].sum()
                metrics.avg_iv_puts = (puts_with_iv['impliedVolatility'] * weights).sum()

        # IV Skew (put IV - call IV, positive means fear/bearish)
        if metrics.avg_iv_calls and metrics.avg_iv_puts:
            metrics.iv_skew = metrics.avg_iv_puts - metrics.avg_iv_calls

        # ATM IV (closest to current price)
        atm_range = current_price * 0.02  # Within 2% of current price
        atm_calls = calls[
            (calls['strike'] >= current_price - atm_range) &
            (calls['strike'] <= current_price + atm_range)
        ]
        atm_puts = puts[
            (puts['strike'] >= current_price - atm_range) &
            (puts['strike'] <= current_price + atm_range)
        ]

        atm_ivs = []
        if len(atm_calls) > 0:
            atm_ivs.extend(atm_calls['impliedVolatility'].dropna().tolist())
        if len(atm_puts) > 0:
            atm_ivs.extend(atm_puts['impliedVolatility'].dropna().tolist())

        if atm_ivs:
            metrics.atm_iv = sum(atm_ivs) / len(atm_ivs)

        # === Unusual Activity Detection ===
        # Look for contracts with volume >> open interest
        metrics.unusual_calls = self._find_unusual_activity(calls, "call", current_price)
        metrics.unusual_puts = self._find_unusual_activity(puts, "put", current_price)

        # Calculate unusual activity score
        total_unusual = len(metrics.unusual_calls) + len(metrics.unusual_puts)
        if total_unusual > 0:
            # Weight by how unusual the activity is
            call_score = sum(u.get('volume_oi_ratio', 0) for u in metrics.unusual_calls)
            put_score = sum(u.get('volume_oi_ratio', 0) for u in metrics.unusual_puts)

            # Normalize to 0-1 scale
            metrics.unusual_activity_score = min(1.0, (call_score + put_score) / 20)

        return metrics

    def _find_unusual_activity(
        self, df, option_type: str, current_price: float, threshold: float = 3.0
    ) -> list[dict]:
        """Find contracts with unusual volume relative to open interest."""
        unusual = []

        # Only consider contracts with meaningful OI
        active = df[df['openInterest'] >= 100].copy()

        for _, row in active.iterrows():
            volume = row['volume'] if pd.notna(row['volume']) else 0
            oi = row['openInterest']

            if oi > 0 and volume > 0:
                ratio = volume / oi
                if ratio >= threshold:
                    unusual.append({
                        'strike': row['strike'],
                        'type': option_type,
                        'volume': int(volume),
                        'open_interest': int(oi),
                        'volume_oi_ratio': round(ratio, 2),
                        'iv': round(row['impliedVolatility'] * 100, 1) if pd.notna(row['impliedVolatility']) else None,
                        'moneyness': 'ITM' if row['inTheMoney'] else 'OTM',
                    })

        # Sort by volume/OI ratio descending
        unusual.sort(key=lambda x: x['volume_oi_ratio'], reverse=True)
        return unusual[:5]  # Top 5 unusual

    def _calculate_sentiment(
        self, symbol: str, metrics: OptionsMetrics
    ) -> OptionsSentiment:
        """Calculate overall sentiment score from options metrics."""

        if not metrics.has_data:
            return OptionsSentiment(
                symbol=symbol,
                sentiment_score=0.5,  # Neutral
                confidence="none",
                metrics=metrics,
            )

        scores = []
        weights = []

        # === Put/Call Ratio Signal ===
        pc_signal = "neutral"
        if metrics.put_call_volume_ratio is not None:
            pcr = metrics.put_call_volume_ratio
            if pcr < 0.7:
                # Bullish - more calls being bought
                pc_score = 0.7 + (0.7 - pcr) * 0.3  # 0.7 to 1.0
                pc_signal = "bullish"
            elif pcr > 1.0:
                # Bearish - more puts being bought
                pc_score = 0.3 - (pcr - 1.0) * 0.15  # 0.3 down to ~0.1
                pc_score = max(0.1, pc_score)
                pc_signal = "bearish"
            else:
                # Neutral range
                pc_score = 0.5

            scores.append(pc_score)
            weights.append(0.4)  # 40% weight

        # === IV Skew Signal ===
        iv_signal = "normal"
        if metrics.iv_skew is not None:
            skew = metrics.iv_skew
            if skew > 0.05:
                # Puts more expensive - fear/hedging
                skew_score = 0.4 - skew * 2
                skew_score = max(0.2, skew_score)
                iv_signal = "high_iv"
            elif skew < -0.03:
                # Calls more expensive - greed
                skew_score = 0.6 + abs(skew) * 2
                skew_score = min(0.8, skew_score)
                iv_signal = "low_iv"
            else:
                skew_score = 0.5

            scores.append(skew_score)
            weights.append(0.2)  # 20% weight

        # === Unusual Activity Signal ===
        flow_signal = "none"
        if metrics.unusual_calls or metrics.unusual_puts:
            call_activity = len(metrics.unusual_calls)
            put_activity = len(metrics.unusual_puts)

            if call_activity > put_activity + 2:
                flow_score = 0.7
                flow_signal = "bullish_flow"
            elif put_activity > call_activity + 2:
                flow_score = 0.3
                flow_signal = "bearish_flow"
            else:
                flow_score = 0.5

            scores.append(flow_score)
            weights.append(0.2)  # 20% weight

        # === OI-based Put/Call (longer-term positioning) ===
        if metrics.put_call_oi_ratio is not None:
            oi_pcr = metrics.put_call_oi_ratio
            if oi_pcr < 0.8:
                oi_score = 0.6
            elif oi_pcr > 1.2:
                oi_score = 0.4
            else:
                oi_score = 0.5

            scores.append(oi_score)
            weights.append(0.2)  # 20% weight

        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            sentiment_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            sentiment_score = 0.5

        # Determine confidence based on data quality
        total_volume = metrics.total_call_volume + metrics.total_put_volume
        total_oi = metrics.total_call_oi + metrics.total_put_oi

        if total_volume > 100000 and total_oi > 500000:
            confidence = "high"
        elif total_volume > 10000 and total_oi > 50000:
            confidence = "medium"
        elif total_volume > 1000:
            confidence = "low"
        else:
            confidence = "none"

        return OptionsSentiment(
            symbol=symbol,
            sentiment_score=sentiment_score,
            confidence=confidence,
            metrics=metrics,
            put_call_signal=pc_signal,
            iv_signal=iv_signal,
            unusual_activity_signal=flow_signal,
        )

    async def get_options_flow_summary(
        self, symbols: list[str]
    ) -> dict[str, OptionsSentiment]:
        """Get options sentiment for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to OptionsSentiment
        """
        tasks = [self.get_options_sentiment(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summary = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, OptionsSentiment):
                summary[symbol] = result
            else:
                # Error case - return neutral
                summary[symbol] = OptionsSentiment(
                    symbol=symbol,
                    sentiment_score=0.5,
                    confidence="none",
                )

        return summary


# Import pandas for type checking in _find_unusual_activity
import pandas as pd
