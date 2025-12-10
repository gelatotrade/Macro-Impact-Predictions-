#!/usr/bin/env python3
"""
Basic Usage Example

Demonstrates how to use the Macro Event Impact Prediction System
to get predictions for upcoming economic events.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.prediction_engine import PredictionEngine
from src.data.economic_calendar import EconomicCalendar, EventImpact
from src.data.market_data_fetcher import MarketDataFetcher
from src.visualization.market_charts import MarketCharts


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("MACRO EVENT IMPACT PREDICTION SYSTEM - Basic Example")
    print("=" * 60)

    # 1. Initialize the prediction engine
    print("\n1. Initializing prediction engine...")
    engine = PredictionEngine()

    # 2. Get current market-implied expectations
    print("\n2. Fetching market-implied expectations...")
    market_exp = engine.get_market_implied_expectations()

    print("\n   Current Market Expectations:")
    print(f"   • VIX: {market_exp.get('vix_current', 'N/A'):.1f}")
    print(f"   • Daily SPX Move: ±{market_exp.get('daily_expected_move_spx', 'N/A'):.2f}%")
    print(f"   • Fed Expectation: {market_exp.get('fed_next_meeting', 'N/A')}")
    print(f"   • P(Cut): {market_exp.get('probability_cut', 0)*100:.0f}%")
    print(f"   • Yield Curve: {'INVERTED' if market_exp.get('curve_inverted') else 'Normal'}")
    print(f"   • Risk Regime: {market_exp.get('risk_regime', 'N/A')}")

    # 3. Get upcoming high-impact events
    print("\n3. Getting upcoming events...")
    calendar = EconomicCalendar()
    events = calendar.get_upcoming_events(days_ahead=7, min_impact=EventImpact.HIGH)

    print(f"\n   Found {len(events)} high-impact events in next 7 days:")
    for event in events[:5]:
        print(f"   • {event.event_name} on {event.date.strftime('%m/%d %H:%M')} ({event.impact.name})")

    # 4. Generate predictions
    print("\n4. Generating predictions...")
    predictions = engine.get_upcoming_predictions(days_ahead=7, min_impact=EventImpact.HIGH)

    print(f"\n   Generated {len(predictions)} predictions:")
    for pred in predictions[:3]:
        spy = pred.expected_moves.get('SPY')
        if spy:
            print(f"\n   📊 {pred.event.event_name}")
            print(f"      Expected SPY Move: ±{spy.expected_move_pct:.2f}%")
            print(f"      1σ Move: ±{spy.move_1sd_pct:.2f}%")
            print(f"      P(Up): {spy.probability_up*100:.0f}% | P(Down): {spy.probability_down*100:.0f}%")

    # 5. Get prediction for a specific event
    print("\n5. Detailed prediction for next event...")
    if predictions:
        pred = predictions[0]
        print(f"\n   Event: {pred.event.event_name}")
        print(f"   Date: {pred.event.date.strftime('%Y-%m-%d %H:%M')} ET")
        print(f"\n   Summary:\n{pred.summary}")

        # Scenario analysis
        print("\n   Scenario Analysis (SPY moves):")
        for scenario, data in pred.scenario_analysis.items():
            spy_move = data['moves'].get('SPY', 0)
            print(f"   • {scenario:15} → {spy_move:+.2f}%")

    # 6. Create visualization
    print("\n6. Creating visualization...")
    charts = MarketCharts()

    if predictions:
        fig = charts.create_expected_moves_chart(predictions[0])
        path = charts.save_chart(fig, "example_prediction")
        print(f"   Chart saved to: {path}")

        # Multi-event chart
        if len(predictions) > 1:
            fig = charts.create_multi_event_prediction_chart(predictions)
            path = charts.save_chart(fig, "example_multi_event")
            print(f"   Chart saved to: {path}")

    print("\n" + "=" * 60)
    print("Example complete! Run 'python main.py dashboard' for interactive view.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
