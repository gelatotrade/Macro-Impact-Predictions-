"""
Impact Analyzer Module

Analyzes historical impact of macro events on various markets.
Used to calibrate prediction models.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from ..data.macro_data_fetcher import MacroDataFetcher
from ..data.market_data_fetcher import MarketDataFetcher


class ImpactAnalyzer:
    """Analyzes the historical impact of macro events on markets."""

    # Historical average reactions (calibrated from real data patterns)
    # Format: {event_type: {instrument: {direction: avg_move_pct}}}
    HISTORICAL_REACTIONS = {
        'cpi': {
            'SPY': {'higher': -0.65, 'lower': 0.55, 'inline': 0.05},
            'QQQ': {'higher': -0.85, 'lower': 0.70, 'inline': 0.08},
            'TLT': {'higher': -0.90, 'lower': 0.75, 'inline': 0.10},
            'DXY': {'higher': 0.35, 'lower': -0.30, 'inline': 0.02},
            'EURUSD': {'higher': -0.30, 'lower': 0.25, 'inline': -0.02},
            'GC=F': {'higher': -0.50, 'lower': 0.45, 'inline': 0.05},
        },
        'nfp': {
            'SPY': {'higher': 0.35, 'lower': -0.25, 'inline': 0.10},
            'QQQ': {'higher': 0.40, 'lower': -0.30, 'inline': 0.12},
            'TLT': {'higher': -0.45, 'lower': 0.40, 'inline': 0.05},
            'DXY': {'higher': 0.40, 'lower': -0.35, 'inline': 0.03},
            'EURUSD': {'higher': -0.35, 'lower': 0.30, 'inline': -0.03},
        },
        'fomc': {
            'SPY': {'hawkish': -0.80, 'dovish': 0.90, 'neutral': 0.20},
            'QQQ': {'hawkish': -1.10, 'dovish': 1.20, 'neutral': 0.25},
            'TLT': {'hawkish': -1.20, 'dovish': 1.40, 'neutral': 0.15},
            'DXY': {'hawkish': 0.50, 'dovish': -0.55, 'neutral': 0.05},
            'EURUSD': {'hawkish': -0.45, 'dovish': 0.50, 'neutral': -0.05},
            'GC=F': {'hawkish': -0.70, 'dovish': 0.80, 'neutral': 0.10},
        },
        'pmi': {
            'SPY': {'higher': 0.30, 'lower': -0.25, 'inline': 0.05},
            'QQQ': {'higher': 0.35, 'lower': -0.30, 'inline': 0.08},
            'DXY': {'higher': 0.20, 'lower': -0.15, 'inline': 0.02},
        },
        'gdp': {
            'SPY': {'higher': 0.40, 'lower': -0.35, 'inline': 0.10},
            'QQQ': {'higher': 0.45, 'lower': -0.40, 'inline': 0.12},
            'TLT': {'higher': -0.35, 'lower': 0.40, 'inline': 0.08},
            'DXY': {'higher': 0.25, 'lower': -0.20, 'inline': 0.03},
        },
        'pce': {
            'SPY': {'higher': -0.50, 'lower': 0.45, 'inline': 0.05},
            'QQQ': {'higher': -0.65, 'lower': 0.55, 'inline': 0.08},
            'TLT': {'higher': -0.70, 'lower': 0.60, 'inline': 0.10},
            'DXY': {'higher': 0.30, 'lower': -0.25, 'inline': 0.02},
        },
        'retail': {
            'SPY': {'higher': 0.25, 'lower': -0.20, 'inline': 0.05},
            'QQQ': {'higher': 0.30, 'lower': -0.25, 'inline': 0.08},
        },
        'claims': {
            'SPY': {'higher': -0.15, 'lower': 0.10, 'inline': 0.02},
            'TLT': {'higher': 0.20, 'lower': -0.15, 'inline': 0.03},
        }
    }

    # Volatility regime multipliers
    VOLATILITY_MULTIPLIERS = {
        'low': 0.7,      # VIX < 15
        'normal': 1.0,   # VIX 15-25
        'high': 1.5,     # VIX 25-35
        'extreme': 2.0   # VIX > 35
    }

    def __init__(self):
        """Initialize the impact analyzer."""
        self.macro_fetcher = MacroDataFetcher()
        self.market_fetcher = MarketDataFetcher()

    def get_historical_reaction(
        self,
        event_type: str,
        instrument: str,
        surprise_direction: str
    ) -> float:
        """Get historical average reaction for an event.

        Args:
            event_type: Type of macro event (cpi, nfp, fomc, etc.)
            instrument: Market instrument symbol
            surprise_direction: 'higher', 'lower', 'inline' (or 'hawkish', 'dovish', 'neutral' for FOMC)

        Returns:
            Expected percentage move
        """
        event_reactions = self.HISTORICAL_REACTIONS.get(event_type.lower(), {})
        instrument_reactions = event_reactions.get(instrument, {})
        return instrument_reactions.get(surprise_direction, 0.0)

    def calculate_event_window_returns(
        self,
        symbol: str,
        event_dates: List[datetime],
        pre_window: int = 1,
        post_window: int = 1
    ) -> pd.DataFrame:
        """Calculate returns around historical events.

        Args:
            symbol: Instrument symbol
            event_dates: List of event dates
            pre_window: Days before event
            post_window: Days after event

        Returns:
            DataFrame with returns for each event
        """
        price_data = self.market_fetcher.fetch_price_data(symbol, period='5y')

        if price_data.empty:
            return pd.DataFrame()

        results = []
        for event_date in event_dates:
            try:
                # Find closest trading day
                idx = price_data.index.get_indexer([event_date], method='nearest')[0]

                if idx < pre_window or idx >= len(price_data) - post_window:
                    continue

                pre_close = price_data['close'].iloc[idx - pre_window]
                event_close = price_data['close'].iloc[idx]
                post_close = price_data['close'].iloc[idx + post_window]

                results.append({
                    'event_date': event_date,
                    'pre_close': pre_close,
                    'event_close': event_close,
                    'post_close': post_close,
                    'event_day_return': (event_close / pre_close - 1) * 100,
                    'post_event_return': (post_close / event_close - 1) * 100,
                    'total_return': (post_close / pre_close - 1) * 100
                })

            except Exception as e:
                logger.debug(f"Error processing event {event_date}: {e}")
                continue

        return pd.DataFrame(results)

    def analyze_surprise_impact(
        self,
        event_type: str,
        surprises: List[float],
        returns: List[float]
    ) -> Dict[str, Any]:
        """Analyze the relationship between surprise magnitude and market returns.

        Args:
            event_type: Type of macro event
            surprises: List of surprise values (actual - consensus)
            returns: List of corresponding market returns

        Returns:
            Statistical analysis results
        """
        if len(surprises) < 10:
            return {'error': 'Insufficient data for analysis'}

        surprises = np.array(surprises)
        returns = np.array(returns)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(surprises, returns)

        # Correlation
        correlation = np.corrcoef(surprises, returns)[0, 1]

        # Categorize surprises
        positive_surprise = surprises > 0
        negative_surprise = surprises < 0

        avg_return_positive = returns[positive_surprise].mean() if positive_surprise.any() else 0
        avg_return_negative = returns[negative_surprise].mean() if negative_surprise.any() else 0

        return {
            'event_type': event_type,
            'n_observations': len(surprises),
            'correlation': correlation,
            'r_squared': r_value ** 2,
            'beta': slope,
            'p_value': p_value,
            'avg_return_positive_surprise': avg_return_positive,
            'avg_return_negative_surprise': avg_return_negative,
            'significant': p_value < 0.05
        }

    def get_volatility_adjusted_impact(
        self,
        event_type: str,
        instrument: str,
        surprise_direction: str,
        current_vix: float
    ) -> Dict[str, float]:
        """Get volatility-adjusted expected impact.

        Args:
            event_type: Type of event
            instrument: Market instrument
            surprise_direction: Direction of surprise
            current_vix: Current VIX level

        Returns:
            Adjusted expected moves
        """
        base_move = self.get_historical_reaction(event_type, instrument, surprise_direction)

        # Determine volatility regime
        if current_vix < 15:
            regime = 'low'
        elif current_vix < 25:
            regime = 'normal'
        elif current_vix < 35:
            regime = 'high'
        else:
            regime = 'extreme'

        multiplier = self.VOLATILITY_MULTIPLIERS[regime]
        adjusted_move = base_move * multiplier

        return {
            'base_expected_move': base_move,
            'volatility_regime': regime,
            'volatility_multiplier': multiplier,
            'adjusted_expected_move': adjusted_move,
            'move_1sd': adjusted_move,
            'move_2sd': adjusted_move * 2
        }

    def get_correlation_matrix(
        self,
        symbols: List[str],
        period: str = '1y'
    ) -> pd.DataFrame:
        """Calculate correlation matrix for instruments.

        Args:
            symbols: List of symbols
            period: Lookback period

        Returns:
            Correlation matrix DataFrame
        """
        price_data = self.market_fetcher.fetch_multiple_symbols(symbols, period=period)

        if not price_data:
            return pd.DataFrame()

        # Calculate returns
        returns_dict = {}
        for symbol, df in price_data.items():
            if not df.empty:
                returns_dict[symbol] = df['close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()

    def identify_regime(self, lookback_days: int = 20) -> Dict[str, Any]:
        """Identify current market regime.

        Args:
            lookback_days: Days to look back for regime identification

        Returns:
            Regime characteristics
        """
        # Get VIX data for volatility regime
        vix_data = self.market_fetcher.fetch_price_data('^VIX', period='6mo')
        spy_data = self.market_fetcher.fetch_price_data('SPY', period='6mo')

        if vix_data.empty or spy_data.empty:
            return {'error': 'Insufficient data'}

        current_vix = vix_data['close'].iloc[-1]
        vix_avg = vix_data['close'].tail(lookback_days).mean()
        vix_percentile = (vix_data['close'] < current_vix).mean() * 100

        # SPY trend
        spy_returns = spy_data['close'].pct_change()
        spy_20d_return = spy_data['close'].iloc[-1] / spy_data['close'].iloc[-lookback_days] - 1
        spy_volatility = spy_returns.tail(lookback_days).std() * np.sqrt(252) * 100

        # Determine regimes
        vol_regime = 'low' if current_vix < 15 else 'normal' if current_vix < 25 else 'high' if current_vix < 35 else 'extreme'
        trend_regime = 'bullish' if spy_20d_return > 0.03 else 'bearish' if spy_20d_return < -0.03 else 'neutral'

        return {
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime,
            'current_vix': current_vix,
            'vix_20d_avg': vix_avg,
            'vix_percentile': vix_percentile,
            'spy_20d_return_pct': spy_20d_return * 100,
            'spy_realized_vol': spy_volatility,
            'regime_summary': f"{trend_regime}_{vol_regime}_volatility"
        }

    def estimate_impact_probability(
        self,
        event_type: str,
        instrument: str,
        expected_surprise: str,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """Estimate probability distribution of impact using Monte Carlo.

        Args:
            event_type: Type of macro event
            instrument: Market instrument
            expected_surprise: Expected surprise direction
            n_simulations: Number of simulations

        Returns:
            Probability distribution metrics
        """
        base_move = self.get_historical_reaction(event_type, instrument, expected_surprise)

        # Assume returns are normally distributed with historical std
        # Using typical event day volatility (1.5x normal daily vol)
        daily_vol = 1.2  # Approximate daily vol in %
        event_vol = daily_vol * 1.5

        # Generate simulations
        np.random.seed(42)
        simulated_returns = np.random.normal(base_move, event_vol, n_simulations)

        return {
            'mean': np.mean(simulated_returns),
            'median': np.median(simulated_returns),
            'std': np.std(simulated_returns),
            'percentile_5': np.percentile(simulated_returns, 5),
            'percentile_25': np.percentile(simulated_returns, 25),
            'percentile_75': np.percentile(simulated_returns, 75),
            'percentile_95': np.percentile(simulated_returns, 95),
            'prob_positive': (simulated_returns > 0).mean(),
            'prob_negative': (simulated_returns < 0).mean(),
            'prob_large_move': (np.abs(simulated_returns) > 1.0).mean(),
            'expected_move': base_move,
            'upside_risk': np.percentile(simulated_returns, 95) - base_move,
            'downside_risk': base_move - np.percentile(simulated_returns, 5)
        }
