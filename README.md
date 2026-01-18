# Stock/Crypto Recommender

An AI-powered investment recommendation system that combines multiple data sources to provide actionable stock and cryptocurrency picks.

## Features

- **Analyst Consensus**: Aggregates recommendations from 10-15 analysts (configurable) with 6-12 month horizons
- **Prediction Markets**: Integrates with Polymarket and other betting markets for sentiment data
- **News Sentiment**: Analyzes latest news and events for market sentiment
- **Weighted Scoring**: Combines all sources with configurable weights for final recommendations

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first if you haven't:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then run the project:

```bash
# Clone the repository
git clone <repository-url>
cd S

# Run directly (uv handles dependencies automatically)
uv run main.py analyze AAPL --type stock

# Or sync dependencies first
uv sync
uv run main.py picks --type crypto
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd S

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

> **Note**: Replace `uv run main.py` with `python main.py` if using pip installation.

### Analyze a Single Asset

```bash
# Analyze a stock
uv run main.py analyze AAPL --type stock

# Analyze a cryptocurrency
uv run main.py analyze BTC --type crypto

# Get detailed analysis
uv run main.py analyze AAPL --type stock --detailed
```

### Get Top Picks

```bash
# Get top stock picks
uv run main.py picks --type stock --top 10

# Get top crypto picks
uv run main.py picks --type crypto --top 5

# Analyze both stocks and crypto
uv run main.py picks --type both

# Analyze custom symbols
uv run main.py picks --symbols "AAPL,MSFT,GOOGL" --type stock
```

### Market Overview

```bash
# Get market overview with macro sentiment
uv run main.py overview
```

### Search Prediction Markets

```bash
# Search for prediction markets
uv run main.py search "bitcoin price"
```

### View Configuration

```bash
uv run main.py config-show
```

## Configuration

Configuration is managed via `config/settings.yaml`. Key settings include:

### Analyst Settings
```yaml
analyst:
  min_analysts: 10      # Minimum analysts for valid recommendation
  max_analysts: 15      # Maximum analysts to consider
  horizon_months_min: 6 # Minimum investment horizon
  horizon_months_max: 12 # Maximum investment horizon
```

### Recommendation Weights
```yaml
recommendation:
  analyst_weight: 0.5   # 50% weight for analyst consensus
  betting_weight: 0.25  # 25% weight for prediction markets
  news_weight: 0.25     # 25% weight for news sentiment
```

### Environment Variables

You can override settings with environment variables:

```bash
export RECOMMENDER_MIN_ANALYSTS=12
export RECOMMENDER_ANALYST_WEIGHT=0.6
export RECOMMENDER_TOP_N=15
```

## Data Sources

### Analyst Data
- Yahoo Finance API for stock analyst recommendations
- CoinGecko API for cryptocurrency metrics

### Prediction Markets
- Polymarket for prediction market data
- Support for additional betting markets via configuration

### News Sources
- RSS feeds from major financial news outlets
- Google News for targeted searches

## Output Example

```
┌─────────────────────────────────────────────────────┐
│ AAPL (STOCK) Strong Buy                             │
├─────────────────────────────────────────────────────┤
│ Overall Score    78.5%                              │
│ Confidence       HIGH                               │
│ Horizon          6-12 months                        │
│                                                     │
│ Analyst Score    82.0% (high)                       │
│ Betting Markets  71.0% (medium)                     │
│ News Sentiment   68.0% (high)                       │
└─────────────────────────────────────────────────────┘

Supporting Factors:
  • Analyst consensus: Strong Buy (14 analysts)
  • Average target upside: 18.5%
  • Prediction markets: 71% bullish ($250,000 volume)
  • News sentiment: Bullish (12 articles)

Risk Factors:
  • None identified
```

## Project Structure

```
S/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.yaml       # Configuration file
└── src/
    ├── __init__.py
    ├── config.py           # Configuration management
    ├── recommender.py      # Main recommendation engine
    └── data_sources/
        ├── __init__.py
        ├── analyst.py      # Analyst data fetching
        ├── betting.py      # Prediction market integration
        └── news.py         # News aggregation
```

## Extending

### Adding New Betting Markets

1. Add configuration in `config/settings.yaml`
2. Implement fetcher in `src/data_sources/betting.py`

### Adding New News Sources

1. Add RSS feed URL to `config/settings.yaml`
2. News aggregator will automatically include it

## Disclaimer

This tool is for informational purposes only and should not be considered financial advice. Always do your own research and consult with a financial advisor before making investment decisions.

## License

MIT License
