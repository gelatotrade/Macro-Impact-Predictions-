"""
Surprise Calculator Module

Calculates and analyzes economic data surprises vs consensus expectations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


class SurpriseDirection(Enum):
    """Direction of surprise relative to consensus."""
    LARGE_BEAT = "large_beat"      # > 2 std above
    BEAT = "beat"                   # 1-2 std above
    SLIGHT_BEAT = "slight_beat"    # 0.5-1 std above
    INLINE = "inline"               # Within 0.5 std
    SLIGHT_MISS = "slight_miss"    # 0.5-1 std below
    MISS = "miss"                   # 1-2 std below
    LARGE_MISS = "large_miss"      # > 2 std below


@dataclass
class SurpriseResult:
    """Result of a surprise calculation."""
    indicator: str
    actual: float
    consensus: float
    previous: float
    surprise: float
    surprise_pct: float
    z_score: float
    direction: SurpriseDirection
    historical_std: float
    percentile: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'indicator': self.indicator,
            'actual': self.actual,
            'consensus': self.consensus,
            'previous': self.previous,
            'surprise': self.surprise,
            'surprise_pct': self.surprise_pct,
            'z_score': self.z_score,
            'direction': self.direction.value,
            'historical_std': self.historical_std,
            'percentile': self.percentile
        }


class SurpriseCalculator:
    """Calculate and analyze economic data surprises."""

    # Historical standard deviations for key indicators (for z-score calculation)
    HISTORICAL_STDS = {
        'CPI MoM': 0.15,
        'CPI YoY': 0.3,
        'Core CPI MoM': 0.1,
        'Core CPI YoY': 0.2,
        'NFP': 80,  # thousands
        'Unemployment Rate': 0.2,
        'ISM Manufacturing PMI': 2.0,
        'ISM Services PMI': 2.0,
        'GDP QoQ': 0.5,
        'Core PCE MoM': 0.1,
        'Core PCE YoY': 0.2,
        'Retail Sales MoM': 0.5,
        'Initial Claims': 15,  # thousands
        'Fed Funds Rate': 0.125,  # one 25bp move
    }

    # Indicator sensitivities (how markets typically react to 1 std surprise)
    SENSITIVITY_MATRIX = {
        'CPI MoM': {
            'SPY': -0.4,
            'QQQ': -0.55,
            'TLT': -0.6,
            'DXY': 0.25,
            'EURUSD': -0.22,
            'GC=F': -0.35
        },
        'CPI YoY': {
            'SPY': -0.5,
            'QQQ': -0.65,
            'TLT': -0.7,
            'DXY': 0.30,
            'EURUSD': -0.25,
            'GC=F': -0.40
        },
        'Core CPI MoM': {
            'SPY': -0.45,
            'QQQ': -0.60,
            'TLT': -0.65,
            'DXY': 0.28,
            'EURUSD': -0.24
        },
        'NFP': {
            'SPY': 0.25,
            'QQQ': 0.30,
            'TLT': -0.35,
            'DXY': 0.30,
            'EURUSD': -0.25
        },
        'Unemployment Rate': {
            'SPY': -0.20,
            'QQQ': -0.25,
            'TLT': 0.30,
            'DXY': -0.20,
            'EURUSD': 0.18
        },
        'ISM Manufacturing PMI': {
            'SPY': 0.20,
            'QQQ': 0.22,
            'DXY': 0.15
        },
        'Fed Funds Rate': {
            'SPY': -0.60,
            'QQQ': -0.80,
            'TLT': -0.90,
            'DXY': 0.45,
            'EURUSD': -0.40,
            'GC=F': -0.55
        },
        'Core PCE MoM': {
            'SPY': -0.35,
            'QQQ': -0.45,
            'TLT': -0.50,
            'DXY': 0.22
        }
    }

    def __init__(self):
        """Initialize the surprise calculator."""
        self._historical_surprises: Dict[str, List[float]] = {}

    def calculate_surprise(
        self,
        indicator: str,
        actual: float,
        consensus: float,
        previous: Optional[float] = None
    ) -> SurpriseResult:
        """Calculate surprise metrics for an economic release.

        Args:
            indicator: Name of the indicator
            actual: Actual released value
            consensus: Consensus expectation
            previous: Previous period value

        Returns:
            SurpriseResult with all metrics
        """
        # Calculate basic surprise
        surprise = actual - consensus
        surprise_pct = (surprise / abs(consensus) * 100) if consensus != 0 else 0

        # Get historical std for z-score
        historical_std = self.HISTORICAL_STDS.get(indicator, abs(consensus) * 0.05)

        # Calculate z-score
        z_score = surprise / historical_std if historical_std > 0 else 0

        # Determine direction
        direction = self._classify_direction(z_score)

        # Calculate percentile (assuming normal distribution)
        percentile = stats.norm.cdf(z_score) * 100

        return SurpriseResult(
            indicator=indicator,
            actual=actual,
            consensus=consensus,
            previous=previous if previous is not None else consensus,
            surprise=surprise,
            surprise_pct=surprise_pct,
            z_score=z_score,
            direction=direction,
            historical_std=historical_std,
            percentile=percentile
        )

    def _classify_direction(self, z_score: float) -> SurpriseDirection:
        """Classify surprise direction based on z-score."""
        if z_score > 2.0:
            return SurpriseDirection.LARGE_BEAT
        elif z_score > 1.0:
            return SurpriseDirection.BEAT
        elif z_score > 0.5:
            return SurpriseDirection.SLIGHT_BEAT
        elif z_score > -0.5:
            return SurpriseDirection.INLINE
        elif z_score > -1.0:
            return SurpriseDirection.SLIGHT_MISS
        elif z_score > -2.0:
            return SurpriseDirection.MISS
        else:
            return SurpriseDirection.LARGE_MISS

    def get_expected_market_reaction(
        self,
        indicator: str,
        z_score: float
    ) -> Dict[str, float]:
        """Get expected market reaction based on surprise z-score.

        Args:
            indicator: Name of the indicator
            z_score: Standardized surprise

        Returns:
            Dictionary of expected moves by instrument
        """
        sensitivities = self.SENSITIVITY_MATRIX.get(indicator, {})

        expected_moves = {}
        for instrument, sensitivity in sensitivities.items():
            # Expected move = sensitivity * z-score
            expected_moves[instrument] = sensitivity * z_score

        return expected_moves

    def calculate_combined_surprise_impact(
        self,
        surprises: List[SurpriseResult]
    ) -> Dict[str, float]:
        """Calculate combined market impact from multiple simultaneous surprises.

        Args:
            surprises: List of SurpriseResult objects

        Returns:
            Combined expected moves by instrument
        """
        combined_moves = {}

        for surprise in surprises:
            moves = self.get_expected_market_reaction(surprise.indicator, surprise.z_score)

            for instrument, move in moves.items():
                if instrument in combined_moves:
                    combined_moves[instrument] += move
                else:
                    combined_moves[instrument] = move

        return combined_moves

    def estimate_scenario_outcomes(
        self,
        indicator: str,
        consensus: float,
        scenarios: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Estimate market outcomes for different release scenarios.

        Args:
            indicator: Name of the indicator
            consensus: Consensus expectation
            scenarios: Optional custom scenarios (beat, miss, etc.)

        Returns:
            Expected outcomes for each scenario
        """
        historical_std = self.HISTORICAL_STDS.get(indicator, abs(consensus) * 0.05)

        if scenarios is None:
            # Generate default scenarios
            scenarios = {
                'large_beat': consensus + 2 * historical_std,
                'beat': consensus + historical_std,
                'slight_beat': consensus + 0.5 * historical_std,
                'inline': consensus,
                'slight_miss': consensus - 0.5 * historical_std,
                'miss': consensus - historical_std,
                'large_miss': consensus - 2 * historical_std
            }

        outcomes = {}
        for scenario_name, actual_value in scenarios.items():
            surprise_result = self.calculate_surprise(indicator, actual_value, consensus)
            expected_moves = self.get_expected_market_reaction(indicator, surprise_result.z_score)

            outcomes[scenario_name] = {
                'actual': actual_value,
                'surprise': surprise_result.surprise,
                'z_score': surprise_result.z_score,
                'direction': surprise_result.direction.value,
                'expected_moves': expected_moves
            }

        return outcomes

    def get_market_priced_expectation(
        self,
        indicator: str,
        current_market_moves: Dict[str, float],
        consensus: float
    ) -> Dict[str, Any]:
        """Infer what the market is pricing in based on current moves.

        Args:
            indicator: Name of the indicator
            current_market_moves: Current pre-release moves by instrument
            consensus: Consensus expectation

        Returns:
            Inferred market expectation
        """
        sensitivities = self.SENSITIVITY_MATRIX.get(indicator, {})

        if not sensitivities:
            return {'error': 'No sensitivity data for indicator'}

        # Use regression to infer implied z-score from market moves
        implied_z_scores = []

        for instrument, move in current_market_moves.items():
            if instrument in sensitivities and sensitivities[instrument] != 0:
                implied_z = move / sensitivities[instrument]
                implied_z_scores.append(implied_z)

        if not implied_z_scores:
            return {'error': 'No matching instruments'}

        avg_implied_z = np.mean(implied_z_scores)
        historical_std = self.HISTORICAL_STDS.get(indicator, abs(consensus) * 0.05)

        implied_value = consensus + avg_implied_z * historical_std

        return {
            'indicator': indicator,
            'consensus': consensus,
            'implied_value': implied_value,
            'implied_surprise': implied_value - consensus,
            'implied_z_score': avg_implied_z,
            'implied_direction': self._classify_direction(avg_implied_z).value,
            'confidence': 'high' if len(implied_z_scores) >= 3 else 'medium' if len(implied_z_scores) >= 2 else 'low'
        }

    def calculate_event_risk_score(
        self,
        indicators: List[str],
        current_vix: float
    ) -> Dict[str, Any]:
        """Calculate overall event risk score.

        Args:
            indicators: List of indicators being released
            current_vix: Current VIX level

        Returns:
            Event risk assessment
        """
        # Base risk from VIX
        vix_risk = min(current_vix / 20, 2.0)  # Normalize to 0-2 scale

        # Risk from event importance
        high_impact_events = ['CPI MoM', 'CPI YoY', 'Core CPI MoM', 'NFP', 'Fed Funds Rate', 'Core PCE MoM']
        event_risk = sum(1.5 if ind in high_impact_events else 0.5 for ind in indicators)

        total_risk = (vix_risk + event_risk) / 2

        return {
            'total_risk_score': min(total_risk, 10),
            'vix_contribution': vix_risk,
            'event_contribution': event_risk,
            'risk_level': 'extreme' if total_risk > 3 else 'high' if total_risk > 2 else 'medium' if total_risk > 1 else 'low',
            'recommendation': 'reduce_position_size' if total_risk > 2 else 'normal_positioning'
        }
