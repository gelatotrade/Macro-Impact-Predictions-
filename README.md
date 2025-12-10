# Macro Event Impact Prediction System

A real-time system for predicting how macroeconomic events (CPI, NFP, PMI, interest rate decisions) will impact financial markets. **Predictions are derived from current market-implied expectations**, not just historical data.

## Key Features

- **Market-Implied Predictions**: Uses VIX, yield curve, Fed Funds Futures, and TIPS spreads to derive expected moves
- **Real-Time Analysis**: Fetches current market data to generate up-to-date predictions
- **Multi-Asset Coverage**: Predicts impacts on equities, bonds, FX, and commodities
- **Scenario Analysis**: Shows expected moves for different outcome scenarios (beat/miss/inline)
- **Interactive Dashboard**: Visualize predictions with an interactive web dashboard
- **Economic Calendar**: Tracks upcoming high-impact events with consensus estimates

## How It Works

Unlike traditional systems that just show historical reactions, this system derives predictions from **what the market is currently pricing in**:

1. **VIX & Implied Volatility** → Expected move magnitude
2. **Yield Curve Shape** → Rate expectations and recession risk
3. **Fed Funds Futures** → Probability of rate cuts/hikes
4. **TIPS Spreads** → Inflation expectations
5. **Historical Sensitivity** → Directional bias for different surprise outcomes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Macro-Impact-Predictions-.git
cd Macro-Impact-Predictions-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and add your API keys (optional but recommended)
```

### API Keys (Optional)

For best results, get free API keys:
- **FRED API**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key

The system works without API keys using demo/sample data.

## Quick Start

### 1. View Predictions (CLI)

```bash
# Show predictions for next 7 days
python main.py predict

# Show predictions for next 14 days with HTML charts
python main.py predict -d 14 --html

# Quick prediction for next event
python main.py quick
```

### 2. Launch Interactive Dashboard

```bash
python main.py dashboard
```

Then open http://127.0.0.1:8050 in your browser.

### 3. Use as Library

```python
from src.models.prediction_engine import PredictionEngine
from src.data.economic_calendar import EventImpact

# Initialize engine
engine = PredictionEngine()

# Get market-implied expectations
market_exp = engine.get_market_implied_expectations()
print(f"VIX: {market_exp['vix_current']}")
print(f"Daily SPX Move: ±{market_exp['daily_expected_move_spx']:.2f}%")
print(f"Fed Expectation: {market_exp['fed_next_meeting']}")

# Get predictions for upcoming events
predictions = engine.get_upcoming_predictions(
    days_ahead=7,
    min_impact=EventImpact.HIGH
)

for pred in predictions:
    spy_move = pred.expected_moves.get('SPY')
    print(f"\n{pred.event.event_name}:")
    print(f"  Expected SPY Move: ±{spy_move.expected_move_pct:.2f}%")
    print(f"  P(Up): {spy_move.probability_up*100:.0f}%")
```

## Output Example

```
📊 CURRENT MARKET EXPECTATIONS:
   VIX: 18.5 (normal volatility)
   Daily SPX Expected Move: ±1.17%
   Fed Next Meeting: hold (Cut: 35%)
   Yield Curve: Normal
   Risk Regime: neutral

[1] CPI MoM
    Date: 2024-01-11 08:30 ET
    Impact: CRITICAL
    Consensus: 0.3%

    📈 PREDICTED MOVES:
       SPY: ↓ ±0.89% (1σ: 1.75%) | P(Up): 42%
       TLT: ±1.22% (Bonds)
       DXY: ±0.58% (USD)

    🎯 SCENARIO ANALYSIS:
       BEAT         → SPY: -1.75%
       INLINE       → SPY: +0.18%
       MISS         → SPY: +1.75%

    ⚠️  Risk Level: MEDIUM (Score: 5.2/10)

    📌 KEY DRIVERS:
       • VIX at 18.5 (normal volatility)
       • Market pricing 35% cut / 10% hike
```

## Project Structure

```
Macro-Impact-Predictions-/
├── main.py                 # Main CLI entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.yaml      # Configuration file
├── src/
│   ├── data/
│   │   ├── macro_data_fetcher.py    # FRED/macro data
│   │   ├── market_data_fetcher.py   # Market prices & IV
│   │   └── economic_calendar.py     # Event calendar
│   ├── analysis/
│   │   ├── impact_analyzer.py       # Historical impact analysis
│   │   └── surprise_calculator.py   # Surprise metrics
│   ├── models/
│   │   └── prediction_engine.py     # Core prediction engine
│   ├── visualization/
│   │   ├── market_charts.py         # Plotly charts
│   │   └── dashboard.py             # Dash web app
│   └── utils/
│       ├── config_loader.py         # Configuration
│       └── logger.py                # Logging
├── examples/
│   ├── basic_usage.py               # Basic usage example
│   └── scenario_analysis.py         # Scenario analysis example
└── tests/                           # Unit tests
```

## Tracked Events

### High-Impact Events
- **Inflation**: CPI, Core CPI, PCE, Core PCE
- **Employment**: Non-Farm Payrolls, Unemployment Rate, Initial Claims
- **Rates**: FOMC Decisions, Fed Chair Speeches
- **Growth**: GDP, Retail Sales
- **PMI**: ISM Manufacturing, ISM Services

### Predicted Instruments
- **Equities**: SPY, QQQ, IWM, DIA
- **Bonds**: TLT, IEF (Treasury ETFs)
- **FX**: DXY, EUR/USD, USD/JPY, GBP/USD
- **Commodities**: Gold (GC=F)
- **Volatility**: VIX

## How Predictions Are Calculated

### 1. Expected Move Magnitude (from VIX)

```
Daily Expected Move = VIX / √252
Event Move = Daily Move × Event Multiplier × Impact Factor
```

### 2. Directional Bias (from Market Expectations)

The system determines which direction the market is likely to move based on:
- Fed rate expectations (hawkish → equities down, dovish → equities up)
- Inflation expectations (rising → bonds down, falling → bonds up)
- Yield curve shape (inverted → risk-off, steep → risk-on)

### 3. Scenario Analysis

For each event, the system calculates expected moves for:
- **Large Beat**: +2σ above consensus
- **Beat**: +1σ above consensus
- **Inline**: Within ±0.5σ
- **Miss**: -1σ below consensus
- **Large Miss**: -2σ below consensus

## Dashboard Features

- **Market Expectations Panel**: Live VIX, Fed pricing, yield curve
- **Events Table**: Upcoming events with expected moves
- **Prediction Charts**: Visual expected move ranges
- **Scenario Comparison**: Compare beat/miss scenarios
- **Distribution Charts**: Probability distribution of outcomes
- **Risk Assessment**: Event risk scoring

## Disclaimer

This system is for educational and research purposes only. The predictions are based on market-implied data and historical patterns, and should not be considered financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open a GitHub issue.
