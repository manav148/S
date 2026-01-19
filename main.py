#!/usr/bin/env python3
"""Stock/Crypto Recommender CLI."""

import asyncio
import logging
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import config
from src.recommender import (
    Recommendation,
    RecommendationEngine,
    RecommendationType,
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_recommendation_color(rec_type: RecommendationType) -> str:
    """Get color for recommendation type."""
    colors = {
        RecommendationType.STRONG_BUY: "bold green",
        RecommendationType.BUY: "green",
        RecommendationType.HOLD: "yellow",
        RecommendationType.SELL: "red",
        RecommendationType.STRONG_SELL: "bold red",
    }
    return colors.get(rec_type, "white")


def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level."""
    colors = {"high": "green", "medium": "yellow", "low": "red"}
    return colors.get(confidence, "white")


def format_recommendation(rec: Recommendation) -> Panel:
    """Format a recommendation as a rich panel."""
    # Header with symbol and recommendation
    rec_color = get_recommendation_color(rec.recommendation)
    conf_color = get_confidence_color(rec.confidence)

    title = Text()
    title.append(f"{rec.symbol} ", style="bold white")
    title.append(f"({rec.asset_type.upper()}) ", style="dim")
    title.append(rec.recommendation.value, style=rec_color)

    # Score breakdown table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Overall Score", f"{rec.overall_score:.1%}")
    table.add_row(
        "Confidence",
        Text(rec.confidence.upper(), style=conf_color),
    )
    table.add_row("Horizon", f"{rec.horizon_months[0]}-{rec.horizon_months[1]} months")

    # Score breakdown
    if rec.analyst_score:
        table.add_row(
            "Analyst Score",
            f"{rec.analyst_score.score:.1%} ({rec.analyst_score.confidence})",
        )
    if rec.betting_score:
        table.add_row(
            "Betting Markets",
            f"{rec.betting_score.score:.1%} ({rec.betting_score.confidence})",
        )
    if rec.news_score:
        table.add_row(
            "News Sentiment",
            f"{rec.news_score.score:.1%} ({rec.news_score.confidence})",
        )

    return Panel(table, title=title, border_style=rec_color)


def format_detailed_recommendation(rec: Recommendation) -> None:
    """Print detailed recommendation with all data."""
    console.print()
    console.print(format_recommendation(rec))

    # Supporting factors
    if rec.supporting_factors:
        console.print("\n[bold green]Supporting Factors:[/]")
        for factor in rec.supporting_factors:
            console.print(f"  • {factor}")

    # Risk factors
    if rec.risk_factors:
        console.print("\n[bold red]Risk Factors:[/]")
        for risk in rec.risk_factors:
            console.print(f"  • {risk}")

    # Analyst details
    if rec.analyst_data and rec.analyst_data.recommendations:
        console.print("\n[bold]Analyst Details:[/]")
        dist = rec.analyst_data.rating_distribution
        console.print(
            f"  Strong Buy: {dist.get('STRONG_BUY', 0)} | "
            f"Buy: {dist.get('BUY', 0)} | "
            f"Hold: {dist.get('HOLD', 0)} | "
            f"Sell: {dist.get('SELL', 0)} | "
            f"Strong Sell: {dist.get('STRONG_SELL', 0)}"
        )
        if rec.analyst_data.current_price:
            console.print(f"  Current Price: ${rec.analyst_data.current_price:,.2f}")
        if rec.analyst_data.average_target_price:
            console.print(
                f"  Avg Target: ${rec.analyst_data.average_target_price:,.2f}"
            )

    # Betting market details
    if rec.betting_data and rec.betting_data.markets:
        console.print("\n[bold]Prediction Markets:[/]")
        console.print(f"  Markets Found: {rec.betting_data.market_count}")
        console.print(f"  Total Volume: ${rec.betting_data.total_volume:,.0f}")
        console.print(
            f"  Bullish Probability: {rec.betting_data.average_bullish_probability:.1%}"
        )

        # Show top markets
        for market in rec.betting_data.markets[:3]:
            console.print(f"  • {market.title[:60]}...")
            console.print(f"    Probability: {market.probability:.1%}")

    # News details
    if rec.news_data and rec.news_data.articles:
        console.print("\n[bold]News Sentiment:[/]")
        console.print(f"  Articles Analyzed: {rec.news_data.article_count}")
        console.print(
            f"  Bullish: {rec.news_data.bullish_count} | "
            f"Neutral: {rec.news_data.neutral_count} | "
            f"Bearish: {rec.news_data.bearish_count}"
        )

        # Show top headlines
        console.print("\n  Recent Headlines:")
        for article in rec.news_data.articles[:5]:
            console.print(f"  • {article.title[:70]}...")
            console.print(f"    Source: {article.source}")


def format_summary_table(recommendations: list[Recommendation]) -> Table:
    """Format recommendations as a summary table."""
    table = Table(title="Top Picks", show_header=True, header_style="bold")

    table.add_column("Rank", justify="center", width=4)
    table.add_column("Symbol", style="bold")
    table.add_column("Type")
    table.add_column("Rec", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Analysts", justify="right")
    table.add_column("Markets", justify="right")
    table.add_column("News", justify="right")
    table.add_column("Confidence", justify="center")

    for i, rec in enumerate(recommendations, 1):
        rec_color = get_recommendation_color(rec.recommendation)
        conf_color = get_confidence_color(rec.confidence)

        analyst_count = rec.analyst_data.total_analysts if rec.analyst_data else 0
        analyst_score = f"{rec.analyst_score.score:.0%}" if rec.analyst_score else "-"
        betting_score = f"{rec.betting_score.score:.0%}" if rec.betting_score else "-"
        news_score = f"{rec.news_score.score:.0%}" if rec.news_score else "-"

        # Color scores based on value
        def score_style(score_obj):
            if not score_obj or score_obj.confidence == "none":
                return "dim"
            return "green" if score_obj.score >= 0.6 else "yellow" if score_obj.score >= 0.4 else "red"

        table.add_row(
            str(i),
            rec.symbol,
            rec.asset_type,
            Text(rec.recommendation.value, style=rec_color),
            f"{rec.overall_score:.1%}",
            Text(f"{analyst_score} ({analyst_count})", style=score_style(rec.analyst_score)),
            Text(betting_score, style=score_style(rec.betting_score)),
            Text(news_score, style=score_style(rec.news_score)),
            Text(rec.confidence, style=conf_color),
        )

    return table


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(verbose: bool, debug: bool) -> None:
    """Stock/Crypto Recommender - AI-powered investment recommendations.

    Analyzes stocks and cryptocurrencies using:
    - Analyst consensus (10-15 analysts, configurable)
    - Prediction market sentiment (Polymarket)
    - News sentiment analysis
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)


@cli.command()
@click.argument("symbol")
@click.option(
    "--type",
    "-t",
    "asset_type",
    type=click.Choice(["stock", "crypto"]),
    default="stock",
    help="Asset type",
)
@click.option("--detailed", "-d", is_flag=True, help="Show detailed analysis")
def analyze(symbol: str, asset_type: str, detailed: bool) -> None:
    """Analyze a single stock or cryptocurrency.

    Example: recommend analyze AAPL --type stock
    """
    console.print(f"\n[bold]Analyzing {symbol.upper()}...[/]\n")

    async def run() -> Recommendation:
        engine = RecommendationEngine()
        try:
            return await engine.get_recommendation(symbol.upper(), asset_type)
        finally:
            await engine.close()

    try:
        rec = asyncio.run(run())

        if detailed:
            format_detailed_recommendation(rec)
        else:
            console.print(format_recommendation(rec))

    except Exception as e:
        console.print(f"[red]Error analyzing {symbol}: {e}[/]")
        sys.exit(1)


@cli.command()
@click.option(
    "--type",
    "-t",
    "asset_type",
    type=click.Choice(["stock", "crypto", "both"]),
    default="both",
    help="Asset type to analyze",
)
@click.option(
    "--top", "-n", "top_n", default=10, help="Number of top picks to show"
)
@click.option(
    "--symbols",
    "-s",
    help="Comma-separated list of symbols to analyze (optional - discovers dynamically if not provided)",
)
@click.option("--detailed", "-d", is_flag=True, help="Show detailed analysis")
def picks(
    asset_type: str, top_n: int, symbols: str | None, detailed: bool
) -> None:
    """Get top picks for stocks or crypto.

    Dynamically discovers stocks/crypto based on:
    - Analyst ratings (Strong Buy/Buy consensus)
    - Most actively traded
    - Market momentum

    Example: recommend picks --type stock --top 10
    """
    # Parse custom symbols if provided
    custom_symbols = None
    if symbols:
        custom_symbols = [s.strip().upper() for s in symbols.split(",")]
        if asset_type == "both":
            asset_type = "stock"  # Default to stock for custom symbols

    async def run() -> list[Recommendation]:
        engine = RecommendationEngine()
        all_recs: list[Recommendation] = []

        try:
            if custom_symbols:
                # Use provided symbols
                console.print(f"[dim]Analyzing {len(custom_symbols)} {asset_type}s...[/]")
                recs = await engine.get_top_picks(custom_symbols, asset_type, top_n=top_n)
                all_recs.extend(recs)
            else:
                # Dynamically discover and analyze
                if asset_type in ("stock", "both"):
                    console.print("[dim]Discovering top stocks (analyst picks + most active)...[/]")
                    stock_symbols = await engine.discover_symbols("stock", limit=top_n * 2)
                    console.print(f"[dim]Analyzing {len(stock_symbols)} stocks...[/]")
                    recs = await engine.get_top_picks(stock_symbols, "stock", top_n=top_n)
                    all_recs.extend(recs)

                if asset_type in ("crypto", "both"):
                    console.print("[dim]Discovering top cryptos (market cap + trending)...[/]")
                    crypto_symbols = await engine.discover_symbols("crypto", limit=top_n * 2)
                    console.print(f"[dim]Analyzing {len(crypto_symbols)} cryptos...[/]")
                    recs = await engine.get_top_picks(crypto_symbols, "crypto", top_n=top_n)
                    all_recs.extend(recs)

        finally:
            await engine.close()

        # Sort all recommendations by score
        all_recs.sort(key=lambda x: x.overall_score, reverse=True)
        return all_recs[:top_n]

    try:
        recommendations = asyncio.run(run())

        if not recommendations:
            console.print("[yellow]No recommendations available.[/]")
            return

        console.print()
        console.print(format_summary_table(recommendations))

        if detailed:
            for rec in recommendations:
                format_detailed_recommendation(rec)

    except Exception as e:
        console.print(f"[red]Error getting picks: {e}[/]")
        sys.exit(1)


@cli.command()
def overview() -> None:
    """Get market overview with macro sentiment and trending topics."""
    console.print("\n[bold]Market Overview[/]\n")

    async def run() -> dict[str, Any]:
        engine = RecommendationEngine()
        try:
            return await engine.get_market_overview()
        finally:
            await engine.close()

    try:
        data = asyncio.run(run())

        # Macro sentiment
        macro = data.get("macro_sentiment", {})
        if macro.get("recession_probability") is not None:
            console.print(
                f"Recession Probability: {macro['recession_probability']:.1%}"
            )
        if macro.get("rate_cut_probability") is not None:
            console.print(
                f"Rate Cut Probability: {macro['rate_cut_probability']:.1%}"
            )

        # Trending topics
        trending = data.get("trending_topics", [])
        if trending:
            console.print("\n[bold]Trending Topics:[/]")
            for topic in trending[:10]:
                console.print(f"  • {topic['topic']} ({topic['count']} mentions)")

        # Latest headlines
        headlines = data.get("latest_headlines", [])
        if headlines:
            console.print("\n[bold]Latest Headlines:[/]")
            for h in headlines[:5]:
                console.print(f"  • {h['title']}")
                console.print(f"    [dim]{h['source']}[/]")

    except Exception as e:
        console.print(f"[red]Error getting overview: {e}[/]")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Maximum number of results")
def search(query: str, limit: int) -> None:
    """Search prediction markets for a topic.

    Example: recommend search "bitcoin price"
    """
    console.print(f"\n[bold]Searching for: {query}[/]\n")

    async def run() -> list:
        engine = RecommendationEngine()
        try:
            return await engine.betting_fetcher.search_markets(query, limit)
        finally:
            await engine.close()

    try:
        markets = asyncio.run(run())

        if not markets:
            console.print("[yellow]No markets found.[/]")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Market", width=50)
        table.add_column("Probability", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Source")

        for market in markets:
            table.add_row(
                market.title[:50] + "..." if len(market.title) > 50 else market.title,
                f"{market.probability:.1%}",
                f"${market.volume:,.0f}" if market.volume else "N/A",
                market.source,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error searching: {e}[/]")
        sys.exit(1)


@cli.command()
def config_show() -> None:
    """Show current configuration settings."""
    console.print("\n[bold]Current Configuration[/]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")

    table.add_row("Min Analysts", str(config.analyst_min_count))
    table.add_row("Max Analysts", str(config.analyst_max_count))
    table.add_row("Horizon", f"{config.horizon_months[0]}-{config.horizon_months[1]} months")
    table.add_row("Analyst Weight", f"{config.analyst_weight:.0%}")
    table.add_row("Betting Weight", f"{config.betting_weight:.0%}")
    table.add_row("News Weight", f"{config.news_weight:.0%}")
    table.add_row("Top N Results", str(config.top_n))
    table.add_row("Verbose Output", str(config.verbose))

    console.print(table)


@cli.command()
@click.option(
    "--type",
    "-t",
    "market_type",
    type=click.Choice(["crypto", "stock", "all", "top"]),
    default="all",
    help="Type of markets to show",
)
@click.option("--limit", "-l", default=15, help="Maximum number of markets to show")
@click.option("--search", "-s", help="Search for specific markets")
def polymarket(market_type: str, limit: int, search: str | None) -> None:
    """Browse Polymarket prediction markets.

    Examples:
        recommend polymarket --type crypto
        recommend polymarket --type stock
        recommend polymarket --search "bitcoin"
        recommend polymarket --type top --limit 20
    """
    from src.data_sources.betting import BettingMarketFetcher

    async def run() -> list:
        fetcher = BettingMarketFetcher()
        try:
            if search:
                console.print(f"\n[bold]Searching Polymarket for: {search}[/]\n")
                return await fetcher.search_markets(search, limit)
            elif market_type == "crypto":
                console.print("\n[bold]Crypto-Related Prediction Markets[/]\n")
                return await fetcher.get_crypto_markets(limit)
            elif market_type == "stock":
                console.print("\n[bold]Stock/Market-Related Prediction Markets[/]\n")
                return await fetcher.get_stock_markets(limit)
            elif market_type == "top":
                console.print("\n[bold]Top Prediction Markets by Liquidity[/]\n")
                return await fetcher.get_top_markets(limit)
            else:
                # Default: show both crypto and stock
                console.print("\n[bold]Crypto & Stock Prediction Markets[/]\n")
                crypto = await fetcher.get_crypto_markets(limit // 2)
                stock = await fetcher.get_stock_markets(limit // 2)
                combined = crypto + stock
                combined.sort(key=lambda m: m.liquidity or 0, reverse=True)
                return combined[:limit]
        finally:
            await fetcher.close()

    try:
        markets = asyncio.run(run())

        if not markets:
            console.print("[yellow]No markets found.[/]")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Market", width=55)
        table.add_column("Prob", justify="right", width=6)
        table.add_column("Liquidity", justify="right", width=12)
        table.add_column("End Date", width=12)

        for market in markets:
            # Color probability based on value
            prob = market.probability
            if prob >= 0.7:
                prob_style = "green"
            elif prob <= 0.3:
                prob_style = "red"
            else:
                prob_style = "yellow"

            end_date = market.end_date.strftime("%Y-%m-%d") if market.end_date else "N/A"

            table.add_row(
                market.title[:55] + "..." if len(market.title) > 55 else market.title,
                Text(f"{prob:.0%}", style=prob_style),
                f"${market.liquidity:,.0f}" if market.liquidity else "N/A",
                end_date,
            )

        console.print(table)
        console.print(f"\n[dim]Showing {len(markets)} markets sorted by liquidity[/]")

    except Exception as e:
        console.print(f"[red]Error fetching markets: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
