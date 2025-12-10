#!/usr/bin/env python3
"""
Macro Event Impact Prediction System

Main entry point for the application. Provides CLI interface
for running predictions and launching the dashboard.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.models.prediction_engine import PredictionEngine
from src.data.economic_calendar import EventImpact
from src.visualization.market_charts import MarketCharts


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    """Launch the interactive dashboard."""
    from src.visualization.dashboard import MacroDashboard

    logger.info("Launching Macro Event Impact Predictor Dashboard...")
    dashboard = MacroDashboard(host=host, port=port)
    dashboard.run(debug=debug)


def run_predictions(days_ahead: int = 7, output_format: str = "console"):
    """Run predictions for upcoming events and display results."""
    logger.info(f"Generating predictions for next {days_ahead} days...")

    engine = PredictionEngine()
    predictions = engine.get_upcoming_predictions(
        days_ahead=days_ahead,
        min_impact=EventImpact.HIGH
    )

    if not predictions:
        print("\nNo high-impact events in the specified timeframe.")
        return

    # Get market expectations
    market_exp = engine.get_market_implied_expectations()

    print("\n" + "="*80)
    print("MACRO EVENT IMPACT PREDICTIONS")
    print("Based on Market-Implied Expectations (VIX, Yield Curve, Fed Futures)")
    print("="*80)

    print(f"\n📊 CURRENT MARKET EXPECTATIONS:")
    print(f"   VIX: {market_exp.get('vix_current', 'N/A'):.1f} ({market_exp.get('vix_regime', 'N/A')} volatility)")
    print(f"   Daily SPX Expected Move: ±{market_exp.get('daily_expected_move_spx', 'N/A'):.2f}%")
    print(f"   Fed Next Meeting: {market_exp.get('fed_next_meeting', 'N/A')} (Cut: {market_exp.get('probability_cut', 0)*100:.0f}%)")
    print(f"   Yield Curve: {'INVERTED' if market_exp.get('curve_inverted') else 'Normal'}")
    print(f"   Risk Regime: {market_exp.get('risk_regime', 'N/A')}")

    print("\n" + "-"*80)
    print("UPCOMING EVENTS & PREDICTED IMPACT")
    print("-"*80)

    for i, pred in enumerate(predictions, 1):
        event = pred.event
        spy_move = pred.expected_moves.get('SPY')
        tlt_move = pred.expected_moves.get('TLT')
        dxy_move = pred.expected_moves.get('DXY')

        print(f"\n[{i}] {event.event_name}")
        print(f"    Date: {event.date.strftime('%Y-%m-%d %H:%M')} ET")
        print(f"    Impact: {event.impact.name}")
        print(f"    Consensus: {event.consensus} {event.unit}" if event.consensus else "    Consensus: N/A")

        print(f"\n    📈 PREDICTED MOVES:")
        if spy_move:
            direction_symbol = "↑" if spy_move.probability_up > 0.5 else "↓" if spy_move.probability_up < 0.5 else "→"
            print(f"       SPY: {direction_symbol} ±{spy_move.expected_move_pct:.2f}% (1σ: {spy_move.move_1sd_pct:.2f}%) | P(Up): {spy_move.probability_up*100:.0f}%")
        if tlt_move:
            print(f"       TLT: ±{tlt_move.expected_move_pct:.2f}% (Bonds)")
        if dxy_move:
            print(f"       DXY: ±{dxy_move.expected_move_pct:.2f}% (USD)")

        print(f"\n    🎯 SCENARIO ANALYSIS:")
        scenarios = pred.scenario_analysis
        for scenario_name in ['beat', 'inline', 'miss']:
            if scenario_name in scenarios:
                spy_scenario = scenarios[scenario_name]['moves'].get('SPY', 0)
                print(f"       {scenario_name.upper():12} → SPY: {spy_scenario:+.2f}%")

        risk = pred.risk_assessment
        print(f"\n    ⚠️  Risk Level: {risk.get('risk_level', 'N/A').upper()} (Score: {risk.get('overall_risk', 0):.1f}/10)")

        if spy_move and spy_move.drivers:
            print(f"\n    📌 KEY DRIVERS:")
            for driver in spy_move.drivers[:2]:
                print(f"       • {driver}")

    print("\n" + "="*80)
    print("DISCLAIMER: Predictions based on market-implied data. Not financial advice.")
    print("="*80 + "\n")

    # Generate charts if requested
    if output_format in ["html", "all"]:
        charts = MarketCharts()

        # Generate multi-event chart
        fig = charts.create_multi_event_prediction_chart(predictions)
        path = charts.save_chart(fig, f"predictions_{datetime.now().strftime('%Y%m%d')}")
        print(f"📊 Chart saved to: {path}")

        # Generate individual event charts
        for i, pred in enumerate(predictions[:3]):  # Top 3 events
            fig = charts.create_expected_moves_chart(pred)
            safe_name = pred.event.event_name.replace(' ', '_').replace('/', '_')[:30]
            path = charts.save_chart(fig, f"event_{i+1}_{safe_name}")
            print(f"📊 Chart saved to: {path}")


def quick_prediction(event_type: str = None):
    """Get quick prediction for next event."""
    engine = PredictionEngine()

    pred = engine.get_prediction_for_next_event(event_type)

    if not pred:
        print(f"No upcoming {'events' if not event_type else event_type + ' events'} found.")
        return

    print(f"\n🎯 NEXT EVENT: {pred.event.event_name}")
    print(f"   Date: {pred.event.date.strftime('%Y-%m-%d %H:%M')} ET")
    print(f"\n{pred.summary}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Macro Event Impact Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py dashboard           Launch interactive dashboard
  python main.py predict             Show predictions for next 7 days
  python main.py predict -d 14       Show predictions for next 14 days
  python main.py predict --html      Generate HTML charts
  python main.py quick               Quick prediction for next event
  python main.py quick --type rates  Quick prediction for next rate event
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Launch interactive dashboard')
    dash_parser.add_argument('--host', default='127.0.0.1', help='Host address')
    dash_parser.add_argument('--port', type=int, default=8050, help='Port number')
    dash_parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Generate predictions')
    pred_parser.add_argument('-d', '--days', type=int, default=7, help='Days ahead to look')
    pred_parser.add_argument('--html', action='store_true', help='Generate HTML charts')
    pred_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    # Quick command
    quick_parser = subparsers.add_parser('quick', help='Quick prediction for next event')
    quick_parser.add_argument('--type', choices=['inflation', 'rates', 'employment', 'growth', 'pmi'],
                              help='Filter by event type')

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if getattr(args, 'verbose', False) else "INFO"
    setup_logger(log_level=log_level)

    if args.command == 'dashboard':
        run_dashboard(
            host=args.host,
            port=args.port,
            debug=not args.no_debug
        )
    elif args.command == 'predict':
        output_format = 'html' if args.html else 'console'
        run_predictions(days_ahead=args.days, output_format=output_format)
    elif args.command == 'quick':
        quick_prediction(event_type=args.type)
    else:
        # Default: show predictions
        run_predictions()


if __name__ == "__main__":
    main()
