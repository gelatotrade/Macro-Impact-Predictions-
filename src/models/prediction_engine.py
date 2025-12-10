"""
Prediction Engine

Uses market-implied data (IV, spreads, futures) to predict expected moves
for upcoming macro events. This is the core engine that generates predictions
based on CURRENT market expectations, not historical data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from ..data.market_data_fetcher import MarketDataFetcher
from ..data.economic_calendar import EconomicCalendar, EconomicEvent, EventImpact
from ..analysis.impact_analyzer import ImpactAnalyzer
from ..analysis.surprise_calculator import SurpriseCalculator


class PredictionConfidence(Enum):
    """Confidence level of prediction."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExpectedMove:
    """Expected move for an instrument."""
    symbol: str
    name: str
    expected_move_pct: float
    move_1sd_pct: float
    move_2sd_pct: float
    direction: str  # 'up', 'down', 'neutral'
    probability_up: float
    probability_down: float
    confidence: PredictionConfidence
    drivers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'expected_move_pct': round(self.expected_move_pct, 3),
            'move_1sd_pct': round(self.move_1sd_pct, 3),
            'move_2sd_pct': round(self.move_2sd_pct, 3),
            'direction': self.direction,
            'probability_up': round(self.probability_up, 2),
            'probability_down': round(self.probability_down, 2),
            'confidence': self.confidence.value,
            'drivers': self.drivers
        }


@dataclass
class EventPrediction:
    """Full prediction for an upcoming macro event."""
    event: EconomicEvent
    timestamp: datetime
    expected_moves: Dict[str, ExpectedMove]
    market_expectations: Dict[str, Any]
    scenario_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event': self.event.to_dict(),
            'timestamp': self.timestamp.isoformat(),
            'expected_moves': {k: v.to_dict() for k, v in self.expected_moves.items()},
            'market_expectations': self.market_expectations,
            'scenario_analysis': self.scenario_analysis,
            'risk_assessment': self.risk_assessment,
            'summary': self.summary
        }


class PredictionEngine:
    """
    Core prediction engine that derives expected moves from market-implied data.

    Key features:
    1. Uses VIX/implied volatility for expected move magnitude
    2. Uses yield curve/Fed Funds futures for rate expectations
    3. Uses inflation breakevens for inflation expectations
    4. Combines with historical sensitivity to predict directional bias
    """

    # Instruments to generate predictions for
    PREDICTION_INSTRUMENTS = {
        'SPY': {'name': 'S&P 500', 'type': 'equity'},
        'QQQ': {'name': 'Nasdaq 100', 'type': 'equity'},
        'IWM': {'name': 'Russell 2000', 'type': 'equity'},
        'TLT': {'name': '20+ Year Treasury', 'type': 'bonds'},
        'IEF': {'name': '7-10 Year Treasury', 'type': 'bonds'},
        'DXY': {'name': 'US Dollar Index', 'type': 'fx'},
        'EURUSD': {'name': 'EUR/USD', 'type': 'fx'},
        'USDJPY': {'name': 'USD/JPY', 'type': 'fx'},
        'GC=F': {'name': 'Gold', 'type': 'commodity'},
    }

    # Event type to category mapping
    EVENT_CATEGORIES = {
        'inflation': ['CPI', 'Core CPI', 'PCE', 'Core PCE', 'PPI'],
        'employment': ['NFP', 'Unemployment', 'Claims', 'ADP', 'JOLTS'],
        'rates': ['FOMC', 'Fed', 'ECB', 'BOJ', 'BOE'],
        'growth': ['GDP', 'Retail Sales', 'Industrial Production'],
        'pmi': ['ISM', 'PMI', 'Manufacturing', 'Services'],
    }

    def __init__(self):
        """Initialize the prediction engine."""
        self.market_fetcher = MarketDataFetcher()
        self.calendar = EconomicCalendar()
        self.impact_analyzer = ImpactAnalyzer()
        self.surprise_calculator = SurpriseCalculator()
        self._cache: Dict[str, Any] = {}

    def get_market_implied_expectations(self) -> Dict[str, Any]:
        """Get current market-implied expectations from various instruments.

        This is the KEY function that derives predictions from market data.
        """
        logger.info("Fetching market-implied expectations...")

        # Get comprehensive market expectations
        expectations = self.market_fetcher.get_market_expectations()

        # Extract key metrics
        iv_data = expectations.get('implied_volatility', {})
        yield_data = expectations.get('yield_curve', {})
        fed_data = expectations.get('fed_expectations', {})
        inflation_data = expectations.get('inflation_expectations', {})
        regime = expectations.get('market_regime', {})

        return {
            'timestamp': datetime.now().isoformat(),

            # Volatility expectations
            'vix_current': iv_data.get('vix_current', 18),
            'vix_regime': iv_data.get('vix_regime', 'normal'),
            'daily_expected_move_spx': iv_data.get('daily_expected_move_pct', 1.0),
            'weekly_expected_move_spx': iv_data.get('weekly_expected_move_pct', 2.2),

            # Rate expectations from yield curve
            'curve_inverted': yield_data.get('curve_inverted', False),
            'rate_expectation': yield_data.get('rate_expectation', 'stable'),
            'spread_10y_3m': yield_data.get('spread_10y_3m'),

            # Fed expectations from futures
            'fed_next_meeting': fed_data.get('next_meeting_expectation', 'hold'),
            'probability_cut': fed_data.get('probability_cut_next', 0.3),
            'probability_hike': fed_data.get('probability_hike_next', 0.1),
            'probability_hold': fed_data.get('probability_hold_next', 0.6),
            'expected_rate_12m': fed_data.get('expected_rate_12m'),

            # Inflation expectations from TIPS
            'breakeven_inflation': inflation_data.get('breakeven_inflation_approx', 2.5),
            'inflation_expectation': inflation_data.get('inflation_expectation', 'stable'),

            # Overall regime
            'risk_regime': regime.get('risk_regime', 'neutral'),
            'growth_regime': regime.get('growth_regime', 'expansion'),
        }

    def predict_event_impact(self, event: EconomicEvent) -> EventPrediction:
        """Generate prediction for a specific macro event.

        This is the main prediction function that combines:
        1. Market-implied volatility for move magnitude
        2. Market-implied expectations for directional bias
        3. Historical sensitivities for relative moves across instruments

        Args:
            event: The economic event to predict

        Returns:
            Complete prediction with expected moves for all instruments
        """
        logger.info(f"Generating prediction for: {event.event_name}")

        # Step 1: Get current market expectations
        market_exp = self.get_market_implied_expectations()

        # Step 2: Determine event category and impact level
        event_category = self._get_event_category(event.event_name)
        impact_multiplier = self._get_impact_multiplier(event.impact)

        # Step 3: Calculate base expected move from implied volatility
        base_daily_move = market_exp['daily_expected_move_spx']

        # Step 4: Adjust for event type (some events cause larger moves)
        event_vol_multiplier = self._get_event_volatility_multiplier(event_category)
        adjusted_move = base_daily_move * event_vol_multiplier * impact_multiplier

        # Step 5: Determine directional bias from market expectations
        directional_bias = self._get_directional_bias(event, market_exp)

        # Step 6: Generate expected moves for each instrument
        expected_moves = {}
        for symbol, info in self.PREDICTION_INSTRUMENTS.items():
            expected_moves[symbol] = self._calculate_instrument_expected_move(
                symbol=symbol,
                name=info['name'],
                instrument_type=info['type'],
                event=event,
                event_category=event_category,
                base_move=adjusted_move,
                directional_bias=directional_bias,
                market_exp=market_exp
            )

        # Step 7: Generate scenario analysis
        scenarios = self._generate_scenario_analysis(event, expected_moves, market_exp)

        # Step 8: Calculate risk assessment
        risk = self._calculate_risk_assessment(event, market_exp)

        # Step 9: Generate summary
        summary = self._generate_prediction_summary(event, expected_moves, market_exp, directional_bias)

        return EventPrediction(
            event=event,
            timestamp=datetime.now(),
            expected_moves=expected_moves,
            market_expectations=market_exp,
            scenario_analysis=scenarios,
            risk_assessment=risk,
            summary=summary
        )

    def _get_event_category(self, event_name: str) -> str:
        """Determine the category of an event."""
        event_name_lower = event_name.lower()

        for category, keywords in self.EVENT_CATEGORIES.items():
            if any(kw.lower() in event_name_lower for kw in keywords):
                return category

        return 'other'

    def _get_impact_multiplier(self, impact: EventImpact) -> float:
        """Get multiplier based on event impact level."""
        multipliers = {
            EventImpact.LOW: 0.5,
            EventImpact.MEDIUM: 0.8,
            EventImpact.HIGH: 1.2,
            EventImpact.CRITICAL: 1.8
        }
        return multipliers.get(impact, 1.0)

    def _get_event_volatility_multiplier(self, event_category: str) -> float:
        """Get volatility multiplier for event type."""
        multipliers = {
            'inflation': 1.5,  # CPI etc cause big moves
            'rates': 2.0,     # FOMC causes biggest moves
            'employment': 1.3,  # NFP
            'growth': 1.1,
            'pmi': 0.9,
            'other': 0.7
        }
        return multipliers.get(event_category, 1.0)

    def _get_directional_bias(
        self,
        event: EconomicEvent,
        market_exp: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine directional bias based on market expectations.

        This uses market-implied data to determine which direction
        the market is leaning for different surprise outcomes.
        """
        event_category = self._get_event_category(event.event_name)

        # Get consensus vs what market is pricing in
        consensus = event.consensus
        previous = event.previous

        # Default neutral bias
        bias = {
            'direction': 'neutral',
            'strength': 0.0,
            'equities_if_beat': 0.0,
            'equities_if_miss': 0.0,
            'bonds_if_beat': 0.0,
            'bonds_if_miss': 0.0,
            'fx_if_beat': 0.0,
            'fx_if_miss': 0.0,
        }

        if event_category == 'inflation':
            # Higher inflation = bad for equities/bonds, good for USD
            # Market expects: If Fed more hawkish, inflation is problem
            fed_hawkish = market_exp.get('probability_hike', 0) > market_exp.get('probability_cut', 0)

            bias['equities_if_beat'] = -1.0 if fed_hawkish else -0.7  # Higher inflation = equities down
            bias['equities_if_miss'] = 0.8 if fed_hawkish else 0.6   # Lower inflation = equities up
            bias['bonds_if_beat'] = -1.2   # Higher inflation = bonds down (yields up)
            bias['bonds_if_miss'] = 1.0    # Lower inflation = bonds up (yields down)
            bias['fx_if_beat'] = 0.5       # Higher inflation = USD up (Fed more hawkish)
            bias['fx_if_miss'] = -0.4      # Lower inflation = USD down

            # Determine overall market lean
            if market_exp.get('inflation_expectation') == 'falling':
                bias['direction'] = 'optimistic'
                bias['strength'] = 0.6
            elif market_exp.get('inflation_expectation') == 'rising':
                bias['direction'] = 'cautious'
                bias['strength'] = 0.4

        elif event_category == 'employment':
            # Strong employment = good for equities, but also higher rates
            curve_inverted = market_exp.get('curve_inverted', False)

            if curve_inverted:
                # In recession fear mode, strong employment is relief
                bias['equities_if_beat'] = 0.6
                bias['equities_if_miss'] = -0.8
            else:
                # Normal mode, strong employment = higher rates concern
                bias['equities_if_beat'] = 0.3
                bias['equities_if_miss'] = -0.4

            bias['bonds_if_beat'] = -0.5   # Strong jobs = yields up
            bias['bonds_if_miss'] = 0.6    # Weak jobs = yields down
            bias['fx_if_beat'] = 0.4       # Strong jobs = USD up
            bias['fx_if_miss'] = -0.3

        elif event_category == 'rates':
            # FOMC - most impactful
            prob_cut = market_exp.get('probability_cut', 0.3)
            prob_hike = market_exp.get('probability_hike', 0.1)

            if prob_cut > 0.6:
                bias['direction'] = 'dovish_expected'
                bias['equities_if_beat'] = 1.2   # Dovish = stocks up
                bias['equities_if_miss'] = -1.5  # Hawkish surprise = stocks down
            elif prob_hike > 0.3:
                bias['direction'] = 'hawkish_expected'
                bias['equities_if_beat'] = -0.8
                bias['equities_if_miss'] = 0.5

            bias['bonds_if_beat'] = 1.5 if prob_cut > 0.5 else -0.8
            bias['bonds_if_miss'] = -1.2 if prob_cut > 0.5 else 0.6
            bias['fx_if_beat'] = -0.6 if prob_cut > 0.5 else 0.5
            bias['fx_if_miss'] = 0.5 if prob_cut > 0.5 else -0.4

        elif event_category == 'growth':
            # GDP, retail sales - growth data
            bias['equities_if_beat'] = 0.5
            bias['equities_if_miss'] = -0.6
            bias['bonds_if_beat'] = -0.3
            bias['bonds_if_miss'] = 0.4
            bias['fx_if_beat'] = 0.3
            bias['fx_if_miss'] = -0.2

        return bias

    def _calculate_instrument_expected_move(
        self,
        symbol: str,
        name: str,
        instrument_type: str,
        event: EconomicEvent,
        event_category: str,
        base_move: float,
        directional_bias: Dict[str, Any],
        market_exp: Dict[str, Any]
    ) -> ExpectedMove:
        """Calculate expected move for a specific instrument."""

        # Adjust base move for instrument type
        type_multipliers = {
            'equity': 1.0,
            'bonds': 0.7,  # Bonds typically move less in % terms
            'fx': 0.5,
            'commodity': 0.8
        }
        instrument_move = base_move * type_multipliers.get(instrument_type, 1.0)

        # Get directional bias for this instrument type
        if instrument_type == 'equity':
            beat_move = directional_bias.get('equities_if_beat', 0)
            miss_move = directional_bias.get('equities_if_miss', 0)
        elif instrument_type == 'bonds':
            beat_move = directional_bias.get('bonds_if_beat', 0)
            miss_move = directional_bias.get('bonds_if_miss', 0)
        elif instrument_type == 'fx':
            beat_move = directional_bias.get('fx_if_beat', 0)
            miss_move = directional_bias.get('fx_if_miss', 0)
        else:
            beat_move = 0
            miss_move = 0

        # Calculate expected move (weighted average of scenarios)
        # Assume 50/50 beat/miss if no strong directional signal
        expected_direction = 'up' if beat_move + miss_move > 0 else 'down' if beat_move + miss_move < 0 else 'neutral'

        # Scale the directional bias to the instrument move
        directional_component = (beat_move + miss_move) / 2 * instrument_move * 0.5

        # Expected move is the absolute magnitude
        expected_move = abs(directional_component) if directional_component != 0 else instrument_move * 0.3

        # Probability calculation based on directional bias
        if beat_move > 0 and miss_move > 0:
            prob_up = 0.7
        elif beat_move < 0 and miss_move < 0:
            prob_up = 0.3
        elif beat_move > 0:
            prob_up = 0.55
        elif miss_move > 0:
            prob_up = 0.45
        else:
            prob_up = 0.5

        prob_down = 1 - prob_up

        # Determine drivers
        drivers = self._get_move_drivers(event_category, instrument_type, market_exp)

        # Confidence based on data quality and regime
        confidence = self._determine_confidence(market_exp)

        return ExpectedMove(
            symbol=symbol,
            name=name,
            expected_move_pct=expected_move,
            move_1sd_pct=instrument_move,
            move_2sd_pct=instrument_move * 2,
            direction=expected_direction,
            probability_up=prob_up,
            probability_down=prob_down,
            confidence=confidence,
            drivers=drivers
        )

    def _get_move_drivers(
        self,
        event_category: str,
        instrument_type: str,
        market_exp: Dict[str, Any]
    ) -> List[str]:
        """Get the key drivers for the expected move."""
        drivers = []

        vix = market_exp.get('vix_current', 18)
        drivers.append(f"VIX at {vix:.1f} ({market_exp.get('vix_regime', 'normal')} volatility)")

        if event_category == 'rates':
            prob_cut = market_exp.get('probability_cut', 0)
            prob_hike = market_exp.get('probability_hike', 0)
            drivers.append(f"Market pricing {prob_cut*100:.0f}% cut / {prob_hike*100:.0f}% hike")

        if event_category == 'inflation':
            inflation_exp = market_exp.get('inflation_expectation', 'stable')
            drivers.append(f"Inflation expectations: {inflation_exp}")
            breakeven = market_exp.get('breakeven_inflation', 2.5)
            drivers.append(f"Breakeven inflation: {breakeven:.2f}%")

        if market_exp.get('curve_inverted'):
            drivers.append("Yield curve inverted (recession signal)")

        return drivers

    def _determine_confidence(self, market_exp: Dict[str, Any]) -> PredictionConfidence:
        """Determine prediction confidence based on market conditions."""
        vix = market_exp.get('vix_current', 18)

        # Lower confidence in high volatility regimes
        if vix > 30:
            return PredictionConfidence.LOW
        elif vix > 22:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.HIGH

    def _generate_scenario_analysis(
        self,
        event: EconomicEvent,
        expected_moves: Dict[str, ExpectedMove],
        market_exp: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate scenario analysis for different outcomes."""
        consensus = event.consensus or 0
        previous = event.previous or consensus

        # Generate scenarios based on event type
        scenarios = {
            'large_beat': {
                'description': 'Significantly above consensus (+2σ)',
                'probability': 0.05,
                'moves': {}
            },
            'beat': {
                'description': 'Above consensus (+1σ)',
                'probability': 0.20,
                'moves': {}
            },
            'inline': {
                'description': 'In line with consensus',
                'probability': 0.50,
                'moves': {}
            },
            'miss': {
                'description': 'Below consensus (-1σ)',
                'probability': 0.20,
                'moves': {}
            },
            'large_miss': {
                'description': 'Significantly below consensus (-2σ)',
                'probability': 0.05,
                'moves': {}
            }
        }

        # Calculate moves for each scenario
        event_category = self._get_event_category(event.event_name)

        for scenario_name, scenario in scenarios.items():
            for symbol, move in expected_moves.items():
                base = move.move_1sd_pct

                if scenario_name == 'large_beat':
                    if event_category == 'inflation':
                        # Higher inflation = bad
                        scenario['moves'][symbol] = -base * 2 if move.direction != 'fx' else base * 1.5
                    else:
                        scenario['moves'][symbol] = base * 2

                elif scenario_name == 'beat':
                    if event_category == 'inflation':
                        scenario['moves'][symbol] = -base if move.direction != 'fx' else base * 0.8
                    else:
                        scenario['moves'][symbol] = base

                elif scenario_name == 'inline':
                    scenario['moves'][symbol] = base * 0.2 * (1 if move.probability_up > 0.5 else -1)

                elif scenario_name == 'miss':
                    if event_category == 'inflation':
                        scenario['moves'][symbol] = base if move.direction != 'fx' else -base * 0.8
                    else:
                        scenario['moves'][symbol] = -base

                elif scenario_name == 'large_miss':
                    if event_category == 'inflation':
                        scenario['moves'][symbol] = base * 2 if move.direction != 'fx' else -base * 1.5
                    else:
                        scenario['moves'][symbol] = -base * 2

        return scenarios

    def _calculate_risk_assessment(
        self,
        event: EconomicEvent,
        market_exp: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk assessment for the event."""
        vix = market_exp.get('vix_current', 18)
        impact_score = event.impact.value

        # Base risk from VIX
        vol_risk = vix / 20  # Normalized

        # Event importance risk
        event_risk = impact_score / 2

        # Total risk score
        total_risk = (vol_risk + event_risk) / 2

        return {
            'overall_risk': round(min(total_risk * 3, 10), 1),  # Scale to 0-10
            'volatility_risk': round(vol_risk * 5, 1),
            'event_risk': round(event_risk * 5, 1),
            'risk_level': 'extreme' if total_risk > 2 else 'high' if total_risk > 1.5 else 'medium' if total_risk > 0.8 else 'low',
            'recommendation': self._get_risk_recommendation(total_risk, event)
        }

    def _get_risk_recommendation(self, risk: float, event: EconomicEvent) -> str:
        """Get trading recommendation based on risk level."""
        if risk > 2:
            return "Consider reducing positions before event. High probability of large moves."
        elif risk > 1.5:
            return "Event could cause significant moves. Adjust position sizes accordingly."
        elif risk > 0.8:
            return "Normal event risk. Standard positioning appropriate."
        else:
            return "Low event risk. Market impact likely to be muted."

    def _generate_prediction_summary(
        self,
        event: EconomicEvent,
        expected_moves: Dict[str, ExpectedMove],
        market_exp: Dict[str, Any],
        directional_bias: Dict[str, Any]
    ) -> str:
        """Generate a natural language summary of the prediction."""
        spy_move = expected_moves.get('SPY')
        tlt_move = expected_moves.get('TLT')
        dxy_move = expected_moves.get('DXY')

        summary_parts = [
            f"**{event.event_name}** prediction based on current market-implied expectations:",
            ""
        ]

        # VIX context
        vix = market_exp.get('vix_current', 18)
        summary_parts.append(f"• VIX at {vix:.1f} implies {market_exp.get('daily_expected_move_spx', 1):.2f}% daily SPX move")

        # Main expected moves
        if spy_move:
            summary_parts.append(f"• SPY expected move: ±{spy_move.expected_move_pct:.2f}% (1σ: {spy_move.move_1sd_pct:.2f}%)")
            summary_parts.append(f"  Probability: {spy_move.probability_up*100:.0f}% up / {spy_move.probability_down*100:.0f}% down")

        if tlt_move:
            summary_parts.append(f"• TLT (bonds) expected move: ±{tlt_move.expected_move_pct:.2f}%")

        if dxy_move:
            summary_parts.append(f"• DXY (USD) expected move: ±{dxy_move.expected_move_pct:.2f}%")

        # Market expectations
        summary_parts.append("")
        summary_parts.append("**Market is pricing:**")

        if market_exp.get('fed_next_meeting'):
            prob_cut = market_exp.get('probability_cut', 0)
            summary_parts.append(f"• Fed: {prob_cut*100:.0f}% probability of cut at next meeting")

        if market_exp.get('curve_inverted'):
            summary_parts.append("• Yield curve inverted - recession concerns elevated")

        return "\n".join(summary_parts)

    def get_upcoming_predictions(
        self,
        days_ahead: int = 7,
        min_impact: EventImpact = EventImpact.HIGH
    ) -> List[EventPrediction]:
        """Get predictions for all upcoming high-impact events.

        Args:
            days_ahead: Number of days to look ahead
            min_impact: Minimum impact level

        Returns:
            List of predictions for upcoming events
        """
        events = self.calendar.get_upcoming_events(
            days_ahead=days_ahead,
            min_impact=min_impact
        )

        predictions = []
        for event in events:
            try:
                prediction = self.predict_event_impact(event)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting {event.event_name}: {e}")

        return predictions

    def get_prediction_for_next_event(
        self,
        event_type: Optional[str] = None
    ) -> Optional[EventPrediction]:
        """Get prediction for the next upcoming event.

        Args:
            event_type: Optional filter by event type

        Returns:
            Prediction or None
        """
        event = self.calendar.get_next_event(event_type)

        if event:
            return self.predict_event_impact(event)

        return None
