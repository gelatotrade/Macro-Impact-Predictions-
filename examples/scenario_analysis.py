#!/usr/bin/env python3
"""
Scenario Analysis Example

Demonstrates how to analyze different outcome scenarios
for upcoming macro events using market-implied data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.prediction_engine import PredictionEngine
from src.analysis.surprise_calculator import SurpriseCalculator
from src.visualization.market_charts import MarketCharts


def analyze_cpi_scenarios():
    """Analyze different CPI outcome scenarios."""
    print("\n" + "="*70)
    print("CPI RELEASE SCENARIO ANALYSIS")
    print("Using market-implied data to predict outcomes")
    print("="*70)

    engine = PredictionEngine()
    surprise_calc = SurpriseCalculator()

    # Get market expectations
    market_exp = engine.get_market_implied_expectations()

    print(f"\n📊 CURRENT MARKET CONTEXT:")
    print(f"   VIX: {market_exp.get('vix_current', 18):.1f} ({market_exp.get('vix_regime', 'normal')} volatility)")
    print(f"   Fed Pricing: {market_exp.get('probability_cut', 0)*100:.0f}% cut / {market_exp.get('probability_hike', 0)*100:.0f}% hike")
    print(f"   Inflation Expectations: {market_exp.get('inflation_expectation', 'stable')}")

    # Define CPI scenarios
    consensus = 0.3  # Example: 0.3% MoM
    print(f"\n📋 CONSENSUS: CPI MoM = {consensus}%")

    scenarios = surprise_calc.estimate_scenario_outcomes(
        indicator='CPI MoM',
        consensus=consensus
    )

    print("\n🎯 SCENARIO ANALYSIS:")
    print("-"*70)
    print(f"{'Scenario':<15} {'Actual':<10} {'Surprise':<12} {'SPY':<10} {'TLT':<10} {'DXY':<10}")
    print("-"*70)

    for scenario_name, data in scenarios.items():
        moves = data['expected_moves']
        print(f"{scenario_name:<15} {data['actual']:<10.2f} {data['surprise']:+.3f}        "
              f"{moves.get('SPY', 0):+.2f}%     {moves.get('TLT', 0):+.2f}%     {moves.get('DXY', 0):+.2f}%")

    print("-"*70)

    # Trading recommendations
    print("\n💡 TRADING IMPLICATIONS:")

    # If inflation expectations are elevated
    if market_exp.get('inflation_expectation') == 'rising':
        print("   • Market is pricing in higher inflation")
        print("   • A MISS (lower CPI) could trigger strong rally")
        print("   • A BEAT (higher CPI) is partially priced in")
    else:
        print("   • Inflation expectations are stable/falling")
        print("   • A BEAT (higher CPI) could cause significant selloff")
        print("   • A MISS (lower CPI) would confirm disinflation narrative")

    # Volatility-based sizing
    vix = market_exp.get('vix_current', 18)
    if vix > 25:
        print("   • HIGH VOLATILITY: Consider reducing position sizes")
    elif vix < 15:
        print("   • LOW VOLATILITY: Market may be complacent; watch for large moves")

    return scenarios


def analyze_fomc_scenarios():
    """Analyze FOMC meeting scenarios."""
    print("\n" + "="*70)
    print("FOMC MEETING SCENARIO ANALYSIS")
    print("="*70)

    engine = PredictionEngine()

    market_exp = engine.get_market_implied_expectations()

    prob_cut = market_exp.get('probability_cut', 0.3)
    prob_hold = market_exp.get('probability_hold', 0.6)
    prob_hike = market_exp.get('probability_hike', 0.1)

    print(f"\n📊 FED FUNDS FUTURES PRICING:")
    print(f"   Probability of CUT:  {prob_cut*100:.0f}%")
    print(f"   Probability of HOLD: {prob_hold*100:.0f}%")
    print(f"   Probability of HIKE: {prob_hike*100:.0f}%")

    print("\n🎯 SCENARIO ANALYSIS:")

    # Base expected daily move
    base_move = market_exp.get('daily_expected_move_spx', 1.0)
    fomc_multiplier = 2.0  # FOMC typically 2x daily vol

    scenarios = {
        'Dovish Cut': {
            'probability': prob_cut * 0.4,
            'SPY': base_move * fomc_multiplier * 1.5,
            'TLT': base_move * fomc_multiplier * 2.0,
            'DXY': -base_move * fomc_multiplier * 1.2
        },
        'Expected Cut': {
            'probability': prob_cut * 0.6,
            'SPY': base_move * fomc_multiplier * 0.8,
            'TLT': base_move * fomc_multiplier * 1.2,
            'DXY': -base_move * fomc_multiplier * 0.8
        },
        'Hawkish Hold': {
            'probability': prob_hold * 0.5,
            'SPY': -base_move * fomc_multiplier * 0.6,
            'TLT': -base_move * fomc_multiplier * 0.8,
            'DXY': base_move * fomc_multiplier * 0.6
        },
        'Dovish Hold': {
            'probability': prob_hold * 0.5,
            'SPY': base_move * fomc_multiplier * 0.3,
            'TLT': base_move * fomc_multiplier * 0.5,
            'DXY': -base_move * fomc_multiplier * 0.3
        },
        'Surprise Hike': {
            'probability': prob_hike,
            'SPY': -base_move * fomc_multiplier * 2.5,
            'TLT': -base_move * fomc_multiplier * 3.0,
            'DXY': base_move * fomc_multiplier * 2.0
        }
    }

    print("-"*70)
    print(f"{'Scenario':<18} {'Prob':<8} {'SPY':<12} {'TLT':<12} {'DXY':<12}")
    print("-"*70)

    for scenario, data in scenarios.items():
        print(f"{scenario:<18} {data['probability']*100:>5.1f}%   "
              f"{data['SPY']:+.2f}%       {data['TLT']:+.2f}%       {data['DXY']:+.2f}%")

    print("-"*70)

    # Calculate expected value
    ev_spy = sum(s['probability'] * s['SPY'] for s in scenarios.values())
    ev_tlt = sum(s['probability'] * s['TLT'] for s in scenarios.values())
    ev_dxy = sum(s['probability'] * s['DXY'] for s in scenarios.values())

    print(f"\n📈 EXPECTED VALUE (probability-weighted):")
    print(f"   SPY: {ev_spy:+.2f}%")
    print(f"   TLT: {ev_tlt:+.2f}%")
    print(f"   DXY: {ev_dxy:+.2f}%")

    # Tail risk
    print(f"\n⚠️  TAIL RISK:")
    print(f"   Worst case (Surprise Hike): SPY {scenarios['Surprise Hike']['SPY']:+.2f}%")
    print(f"   Best case (Dovish Cut): SPY {scenarios['Dovish Cut']['SPY']:+.2f}%")

    return scenarios


def main():
    """Run scenario analysis examples."""
    print("\n" + "🎯 " * 20)
    print("MACRO EVENT SCENARIO ANALYZER")
    print("Predictions derived from market-implied data")
    print("🎯 " * 20)

    # CPI Analysis
    cpi_scenarios = analyze_cpi_scenarios()

    # FOMC Analysis
    fomc_scenarios = analyze_fomc_scenarios()

    # Generate charts
    print("\n📊 Generating visualization charts...")
    charts = MarketCharts()

    engine = PredictionEngine()
    predictions = engine.get_upcoming_predictions(days_ahead=14)

    if predictions:
        for i, pred in enumerate(predictions[:2]):
            fig = charts.create_scenario_comparison_chart(pred)
            path = charts.save_chart(fig, f"scenario_analysis_{i+1}")
            print(f"   Saved: {path}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
