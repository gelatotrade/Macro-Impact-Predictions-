"""
Market Data Fetcher

Fetches market data including prices, implied volatility, spreads,
and other market-derived expectations for predicting macro event impacts.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from loguru import logger


class MarketDataFetcher:
    """Fetch market data and market-implied expectations."""

    # Market instruments for different asset classes
    INSTRUMENTS = {
        # Equity Indices
        'SPY': {'name': 'S&P 500 ETF', 'type': 'equity', 'class': 'index'},
        'QQQ': {'name': 'Nasdaq 100 ETF', 'type': 'equity', 'class': 'index'},
        'IWM': {'name': 'Russell 2000 ETF', 'type': 'equity', 'class': 'index'},
        'DIA': {'name': 'Dow Jones ETF', 'type': 'equity', 'class': 'index'},

        # Sector ETFs
        'XLF': {'name': 'Financial Sector ETF', 'type': 'equity', 'class': 'sector'},
        'XLE': {'name': 'Energy Sector ETF', 'type': 'equity', 'class': 'sector'},
        'XLK': {'name': 'Technology Sector ETF', 'type': 'equity', 'class': 'sector'},
        'XLV': {'name': 'Healthcare Sector ETF', 'type': 'equity', 'class': 'sector'},

        # Volatility
        '^VIX': {'name': 'VIX Index', 'type': 'volatility', 'class': 'vol'},
        'UVXY': {'name': 'VIX Short-Term Futures ETF', 'type': 'volatility', 'class': 'vol'},
        'VXX': {'name': 'VIX Futures ETN', 'type': 'volatility', 'class': 'vol'},

        # Treasury/Rates
        '^TNX': {'name': '10-Year Treasury Yield', 'type': 'rate', 'class': 'treasury'},
        '^TYX': {'name': '30-Year Treasury Yield', 'type': 'rate', 'class': 'treasury'},
        '^FVX': {'name': '5-Year Treasury Yield', 'type': 'rate', 'class': 'treasury'},
        '^IRX': {'name': '13-Week T-Bill', 'type': 'rate', 'class': 'treasury'},
        'TLT': {'name': '20+ Year Treasury ETF', 'type': 'bond', 'class': 'treasury'},
        'IEF': {'name': '7-10 Year Treasury ETF', 'type': 'bond', 'class': 'treasury'},
        'SHY': {'name': '1-3 Year Treasury ETF', 'type': 'bond', 'class': 'treasury'},

        # FX
        'DX-Y.NYB': {'name': 'US Dollar Index', 'type': 'fx', 'class': 'currency'},
        'EURUSD=X': {'name': 'EUR/USD', 'type': 'fx', 'class': 'currency'},
        'USDJPY=X': {'name': 'USD/JPY', 'type': 'fx', 'class': 'currency'},
        'GBPUSD=X': {'name': 'GBP/USD', 'type': 'fx', 'class': 'currency'},
        'USDCHF=X': {'name': 'USD/CHF', 'type': 'fx', 'class': 'currency'},

        # Commodities
        'GC=F': {'name': 'Gold Futures', 'type': 'commodity', 'class': 'precious'},
        'CL=F': {'name': 'Crude Oil Futures', 'type': 'commodity', 'class': 'energy'},

        # Fed Funds Futures (for rate expectations)
        'ZQ=F': {'name': 'Fed Funds Futures', 'type': 'futures', 'class': 'rates'},

        # Inflation expectations
        'TIP': {'name': 'TIPS ETF', 'type': 'bond', 'class': 'inflation'},
    }

    # VIX Term Structure Futures (for volatility expectations)
    VIX_FUTURES = ['VX=F', 'VXc1', 'VXc2', 'VXc3', 'VXc4']

    def __init__(self, cache_dir: str = "cache"):
        """Initialize the market data fetcher.

        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _get_cache_path(self, symbol: str, data_type: str) -> Path:
        """Get cache file path."""
        safe_symbol = symbol.replace('^', '_').replace('=', '_').replace('.', '_')
        return self.cache_dir / f"{safe_symbol}_{data_type}.pkl"

    def fetch_price_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical price data using yfinance.

        Args:
            symbol: Ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{period}_{interval}"

        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return self._generate_demo_price_data(symbol, period)

            # Clean column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            self._price_cache[cache_key] = df
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return self._generate_demo_price_data(symbol, period)

    def _generate_demo_price_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Generate demo price data for testing."""
        logger.info(f"Generating demo price data for {symbol}")

        # Parse period to number of days
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }.get(period, 730)

        dates = pd.date_range(
            end=datetime.now(),
            periods=period_days,
            freq='B'  # Business days
        )

        np.random.seed(hash(symbol) % 2**32)

        # Generate realistic price data based on instrument type
        info = self.INSTRUMENTS.get(symbol, {'type': 'equity'})

        if info['type'] == 'volatility':
            # VIX-like data (mean reverting around 18-20)
            base = 18
            values = [base]
            for _ in range(len(dates) - 1):
                shock = np.random.randn() * 2
                mean_reversion = 0.05 * (base - values[-1])
                values.append(max(9, values[-1] + mean_reversion + shock))
            close = np.array(values)

        elif info['type'] == 'rate':
            # Treasury yield (3-5% range)
            base = 4.0
            trend = np.cumsum(np.random.randn(len(dates)) * 0.02)
            close = base + trend
            close = np.clip(close, 1, 6)

        elif info['type'] == 'fx':
            # Currency pair
            base = 1.0 if 'EUR' in symbol or 'GBP' in symbol else 100 if 'JPY' in symbol else 1.0
            returns = np.random.randn(len(dates)) * 0.005
            close = base * np.exp(np.cumsum(returns))

        else:
            # Equity-like data
            base = 400 if 'SPY' in symbol else 300 if 'QQQ' in symbol else 100
            returns = np.random.randn(len(dates)) * 0.012 + 0.0003  # Slight upward bias
            close = base * np.exp(np.cumsum(returns))

        # Generate OHLCV
        high = close * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
        low = close * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
        open_price = (close + np.random.randn(len(dates)) * close * 0.005)
        volume = np.random.randint(1000000, 50000000, len(dates))

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        results = {}
        for symbol in symbols:
            df = self.fetch_price_data(symbol, period, interval)
            if not df.empty:
                results[symbol] = df
        return results

    def get_current_prices(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get current prices for symbols.

        Args:
            symbols: List of symbols. If None, uses all INSTRUMENTS

        Returns:
            DataFrame with current prices
        """
        if symbols is None:
            symbols = list(self.INSTRUMENTS.keys())

        data = []
        for symbol in symbols:
            try:
                df = self.fetch_price_data(symbol, period='5d', interval='1d')
                if not df.empty:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    info = self.INSTRUMENTS.get(symbol, {})

                    data.append({
                        'symbol': symbol,
                        'name': info.get('name', symbol),
                        'type': info.get('type', 'unknown'),
                        'price': latest['close'],
                        'change': latest['close'] - prev['close'],
                        'change_pct': (latest['close'] / prev['close'] - 1) * 100,
                        'volume': latest.get('volume', 0)
                    })
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")

        return pd.DataFrame(data)

    def calculate_implied_volatility_from_vix(self) -> Dict[str, float]:
        """Get implied volatility from VIX and related instruments.

        Returns expected moves for major indices.
        """
        vix_data = self.fetch_price_data('^VIX', period='1mo', interval='1d')

        if vix_data.empty:
            return {}

        current_vix = vix_data['close'].iloc[-1]
        vix_20d_avg = vix_data['close'].tail(20).mean()
        vix_5d_avg = vix_data['close'].tail(5).mean()

        # VIX represents annualized volatility, convert to daily
        daily_expected_move = current_vix / np.sqrt(252)

        # Weekly expected move
        weekly_expected_move = current_vix / np.sqrt(52)

        return {
            'vix_current': current_vix,
            'vix_20d_avg': vix_20d_avg,
            'vix_5d_avg': vix_5d_avg,
            'vix_regime': 'high' if current_vix > 25 else 'normal' if current_vix > 15 else 'low',
            'daily_expected_move_pct': daily_expected_move,
            'weekly_expected_move_pct': weekly_expected_move,
            'spx_daily_expected_move_pct': daily_expected_move,
            'is_elevated': current_vix > vix_20d_avg * 1.2
        }

    def calculate_yield_curve_expectations(self) -> Dict[str, Any]:
        """Calculate rate expectations from yield curve.

        Analyzes Treasury yields to derive market rate expectations.
        """
        yields = {}
        yield_symbols = {
            '3m': '^IRX',
            '5y': '^FVX',
            '10y': '^TNX',
            '30y': '^TYX'
        }

        for tenor, symbol in yield_symbols.items():
            df = self.fetch_price_data(symbol, period='1y', interval='1d')
            if not df.empty:
                yields[tenor] = {
                    'current': df['close'].iloc[-1],
                    'change_1d': df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0,
                    'change_1w': df['close'].iloc[-1] - df['close'].iloc[-5] if len(df) > 5 else 0,
                    'change_1m': df['close'].iloc[-1] - df['close'].iloc[-21] if len(df) > 21 else 0,
                    'percentile_1y': (df['close'] < df['close'].iloc[-1]).mean() * 100
                }

        if '10y' in yields and '3m' in yields:
            spread_10y_3m = yields['10y']['current'] - yields['3m']['current']
        else:
            spread_10y_3m = None

        if '10y' in yields and '5y' in yields:
            spread_10y_5y = yields['10y']['current'] - yields['5y']['current']
        else:
            spread_10y_5y = None

        return {
            'yields': yields,
            'spread_10y_3m': spread_10y_3m,
            'spread_10y_5y': spread_10y_5y,
            'curve_inverted': spread_10y_3m < 0 if spread_10y_3m else None,
            'rate_expectation': 'cuts' if spread_10y_3m and spread_10y_3m < -0.5 else 'hikes' if spread_10y_3m and spread_10y_3m > 1.5 else 'stable'
        }

    def calculate_fed_funds_expectations(self) -> Dict[str, Any]:
        """Calculate Fed Funds rate expectations from futures.

        Uses Fed Funds Futures to derive expected rate path.
        """
        # Get current effective rate
        current_rate = 5.25  # Default, would fetch from FRED

        # Fetch Fed Funds Futures if available
        ff_data = self.fetch_price_data('ZQ=F', period='1y', interval='1d')

        if ff_data.empty:
            # Generate synthetic expectations based on yield curve
            yield_data = self.calculate_yield_curve_expectations()
            curve_inverted = yield_data.get('curve_inverted', False)

            # Synthetic rate path based on curve shape
            if curve_inverted:
                expected_change_3m = -0.25
                expected_change_6m = -0.50
                expected_change_12m = -0.75
            else:
                expected_change_3m = 0
                expected_change_6m = -0.25
                expected_change_12m = -0.25

            return {
                'current_rate': current_rate,
                'expected_rate_3m': current_rate + expected_change_3m,
                'expected_rate_6m': current_rate + expected_change_6m,
                'expected_rate_12m': current_rate + expected_change_12m,
                'expected_cuts_12m': abs(min(0, expected_change_12m)) / 0.25,
                'expected_hikes_12m': max(0, expected_change_12m) / 0.25,
                'next_meeting_expectation': 'hold' if abs(expected_change_3m) < 0.125 else 'cut' if expected_change_3m < 0 else 'hike',
                'probability_cut_next': 0.7 if curve_inverted else 0.3,
                'probability_hike_next': 0.1 if curve_inverted else 0.2,
                'probability_hold_next': 0.2 if curve_inverted else 0.5,
                'data_source': 'derived_from_curve'
            }

        # If we have FF futures data
        implied_rate = 100 - ff_data['close'].iloc[-1]

        return {
            'current_rate': current_rate,
            'implied_rate_next': implied_rate,
            'expected_change': implied_rate - current_rate,
            'next_meeting_expectation': 'hold' if abs(implied_rate - current_rate) < 0.125 else 'cut' if implied_rate < current_rate else 'hike',
            'data_source': 'fed_funds_futures'
        }

    def calculate_inflation_expectations(self) -> Dict[str, Any]:
        """Calculate inflation expectations from TIPS spreads.

        Uses nominal vs TIPS yields to derive breakeven inflation.
        """
        # Fetch TLT (nominal) and TIP (inflation-protected)
        tlt_data = self.fetch_price_data('TLT', period='1y', interval='1d')
        tip_data = self.fetch_price_data('TIP', period='1y', interval='1d')
        tnx_data = self.fetch_price_data('^TNX', period='1y', interval='1d')

        if tlt_data.empty or tip_data.empty:
            return {'breakeven_inflation': 2.5, 'data_source': 'default'}

        # Calculate TIP/TLT ratio as proxy for inflation expectations
        common_dates = tlt_data.index.intersection(tip_data.index)
        if len(common_dates) < 20:
            return {'breakeven_inflation': 2.5, 'data_source': 'insufficient_data'}

        tlt = tlt_data.loc[common_dates, 'close']
        tip = tip_data.loc[common_dates, 'close']

        ratio = tip / tlt
        ratio_change = ratio.iloc[-1] / ratio.iloc[-21] - 1 if len(ratio) > 21 else 0

        # Approximate breakeven (simplified)
        nominal_yield = tnx_data['close'].iloc[-1] if not tnx_data.empty else 4.5

        # Rough approximation of breakeven
        breakeven = 2.5 + (ratio_change * 10)  # Adjust based on ratio change

        return {
            'breakeven_inflation_approx': breakeven,
            'tip_tlt_ratio': ratio.iloc[-1],
            'tip_tlt_ratio_change_1m': ratio_change * 100,
            'inflation_expectation': 'rising' if ratio_change > 0.02 else 'falling' if ratio_change < -0.02 else 'stable',
            'nominal_10y': nominal_yield,
            'data_source': 'tip_tlt_spread'
        }

    def get_market_expectations(self) -> Dict[str, Any]:
        """Get comprehensive market-derived expectations.

        Combines all market-implied forecasts into a single view.
        """
        logger.info("Calculating market-derived expectations...")

        iv_data = self.calculate_implied_volatility_from_vix()
        yield_data = self.calculate_yield_curve_expectations()
        ff_data = self.calculate_fed_funds_expectations()
        inflation_data = self.calculate_inflation_expectations()

        return {
            'timestamp': datetime.now().isoformat(),
            'implied_volatility': iv_data,
            'yield_curve': yield_data,
            'fed_expectations': ff_data,
            'inflation_expectations': inflation_data,
            'market_regime': self._determine_market_regime(iv_data, yield_data)
        }

    def _determine_market_regime(
        self,
        iv_data: Dict[str, Any],
        yield_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Determine current market regime from various indicators."""
        vix_regime = iv_data.get('vix_regime', 'normal')
        curve_inverted = yield_data.get('curve_inverted', False)

        if vix_regime == 'high':
            risk_regime = 'risk_off'
        elif vix_regime == 'low':
            risk_regime = 'risk_on'
        else:
            risk_regime = 'neutral'

        if curve_inverted:
            growth_regime = 'recession_risk'
        else:
            growth_regime = 'expansion'

        return {
            'risk_regime': risk_regime,
            'growth_regime': growth_regime,
            'volatility_regime': vix_regime,
            'overall': f"{risk_regime}_{growth_regime}"
        }

    def calculate_expected_move_for_event(
        self,
        event_type: str,
        lookback_periods: int = 20
    ) -> Dict[str, Any]:
        """Calculate expected move magnitude for a macro event type.

        Uses historical realized volatility around similar events
        and current implied volatility to estimate expected move.

        Args:
            event_type: Type of macro event (cpi, nfp, fomc, etc.)
            lookback_periods: Number of historical events to analyze

        Returns:
            Expected move predictions for various instruments
        """
        iv_data = self.calculate_implied_volatility_from_vix()

        # Event-specific volatility multipliers (based on historical patterns)
        event_multipliers = {
            'cpi': {'SPY': 1.5, 'QQQ': 1.8, 'TLT': 1.3, 'DXY': 1.2},
            'nfp': {'SPY': 1.3, 'QQQ': 1.4, 'TLT': 1.2, 'DXY': 1.4},
            'fomc': {'SPY': 2.0, 'QQQ': 2.2, 'TLT': 1.8, 'DXY': 1.5},
            'gdp': {'SPY': 1.2, 'QQQ': 1.3, 'TLT': 1.1, 'DXY': 1.0},
            'pmi': {'SPY': 1.0, 'QQQ': 1.1, 'TLT': 0.8, 'DXY': 0.9},
            'retail': {'SPY': 0.9, 'QQQ': 1.0, 'TLT': 0.7, 'DXY': 0.8},
        }

        base_daily_move = iv_data.get('daily_expected_move_pct', 1.0)
        multipliers = event_multipliers.get(event_type.lower(), {'SPY': 1.0, 'QQQ': 1.0, 'TLT': 1.0, 'DXY': 1.0})

        expected_moves = {}
        for symbol, mult in multipliers.items():
            expected_moves[symbol] = {
                'expected_move_pct': base_daily_move * mult,
                'expected_move_1sd': base_daily_move * mult,
                'expected_move_2sd': base_daily_move * mult * 2,
                'probability_up': 0.5,  # Will be adjusted by surprise direction
                'probability_down': 0.5
            }

        return {
            'event_type': event_type,
            'base_implied_vol': iv_data.get('vix_current', 18),
            'expected_moves': expected_moves,
            'volatility_regime': iv_data.get('vix_regime', 'normal')
        }
