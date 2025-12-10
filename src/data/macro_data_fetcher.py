"""
Macro Economic Data Fetcher

Fetches macroeconomic data from FRED API and other sources.
Handles CPI, NFP, PMI, Interest Rates, GDP, etc.
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import requests
from loguru import logger


class MacroDataFetcher:
    """Fetch macroeconomic data from FRED and other APIs."""

    # FRED API base URL
    FRED_BASE_URL = "https://api.stlouisfed.org/fred"

    # Key economic indicators with FRED series IDs
    INDICATORS = {
        'cpi': {
            'series_id': 'CPIAUCSL',
            'name': 'Consumer Price Index (All Urban Consumers)',
            'frequency': 'monthly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates'],
            'typical_reaction': {
                'higher_than_expected': {'SPY': -0.5, 'DXY': 0.3, 'TNX': 0.05},
                'lower_than_expected': {'SPY': 0.4, 'DXY': -0.2, 'TNX': -0.04}
            }
        },
        'core_cpi': {
            'series_id': 'CPILFESL',
            'name': 'Core CPI (Ex Food & Energy)',
            'frequency': 'monthly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates'],
            'typical_reaction': {
                'higher_than_expected': {'SPY': -0.6, 'DXY': 0.35, 'TNX': 0.06},
                'lower_than_expected': {'SPY': 0.5, 'DXY': -0.25, 'TNX': -0.05}
            }
        },
        'cpi_yoy': {
            'series_id': 'CPIAUCSL',
            'name': 'CPI Year-over-Year',
            'frequency': 'monthly',
            'impact': 'high',
            'transform': 'yoy_pct',
            'affects': ['equities', 'fx', 'rates']
        },
        'nfp': {
            'series_id': 'PAYEMS',
            'name': 'Non-Farm Payrolls',
            'frequency': 'monthly',
            'impact': 'high',
            'transform': 'diff',
            'affects': ['equities', 'fx', 'rates'],
            'typical_reaction': {
                'higher_than_expected': {'SPY': 0.3, 'DXY': 0.4, 'TNX': 0.03},
                'lower_than_expected': {'SPY': -0.2, 'DXY': -0.3, 'TNX': -0.02}
            }
        },
        'unemployment': {
            'series_id': 'UNRATE',
            'name': 'Unemployment Rate',
            'frequency': 'monthly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates'],
            'typical_reaction': {
                'higher_than_expected': {'SPY': -0.3, 'DXY': -0.2, 'TNX': -0.02},
                'lower_than_expected': {'SPY': 0.2, 'DXY': 0.15, 'TNX': 0.015}
            }
        },
        'fed_funds': {
            'series_id': 'FEDFUNDS',
            'name': 'Federal Funds Effective Rate',
            'frequency': 'monthly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates']
        },
        'fed_funds_target_upper': {
            'series_id': 'DFEDTARU',
            'name': 'Fed Funds Target Range - Upper',
            'frequency': 'daily',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates']
        },
        'ism_pmi': {
            'series_id': 'NAPM',
            'name': 'ISM Manufacturing PMI',
            'frequency': 'monthly',
            'impact': 'medium',
            'affects': ['equities', 'fx'],
            'typical_reaction': {
                'higher_than_expected': {'SPY': 0.25, 'DXY': 0.15},
                'lower_than_expected': {'SPY': -0.2, 'DXY': -0.1}
            }
        },
        'ism_services': {
            'series_id': 'NMFCI',
            'name': 'ISM Non-Manufacturing Index',
            'frequency': 'monthly',
            'impact': 'medium',
            'affects': ['equities', 'fx']
        },
        'gdp': {
            'series_id': 'GDP',
            'name': 'Gross Domestic Product',
            'frequency': 'quarterly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates']
        },
        'gdp_growth': {
            'series_id': 'A191RL1Q225SBEA',
            'name': 'Real GDP Growth Rate',
            'frequency': 'quarterly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates']
        },
        'retail_sales': {
            'series_id': 'RSAFS',
            'name': 'Advance Retail Sales',
            'frequency': 'monthly',
            'impact': 'medium',
            'transform': 'pct_change',
            'affects': ['equities', 'fx']
        },
        'industrial_production': {
            'series_id': 'INDPRO',
            'name': 'Industrial Production Index',
            'frequency': 'monthly',
            'impact': 'medium',
            'affects': ['equities']
        },
        'housing_starts': {
            'series_id': 'HOUST',
            'name': 'Housing Starts',
            'frequency': 'monthly',
            'impact': 'medium',
            'affects': ['equities', 'rates']
        },
        'building_permits': {
            'series_id': 'PERMIT',
            'name': 'Building Permits',
            'frequency': 'monthly',
            'impact': 'low',
            'affects': ['equities']
        },
        'pce': {
            'series_id': 'PCE',
            'name': 'Personal Consumption Expenditures',
            'frequency': 'monthly',
            'impact': 'medium',
            'affects': ['equities', 'rates']
        },
        'core_pce': {
            'series_id': 'PCEPILFE',
            'name': 'Core PCE Price Index',
            'frequency': 'monthly',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates'],
            'typical_reaction': {
                'higher_than_expected': {'SPY': -0.4, 'DXY': 0.25, 'TNX': 0.04},
                'lower_than_expected': {'SPY': 0.35, 'DXY': -0.2, 'TNX': -0.03}
            }
        },
        'initial_claims': {
            'series_id': 'ICSA',
            'name': 'Initial Jobless Claims',
            'frequency': 'weekly',
            'impact': 'medium',
            'affects': ['equities', 'rates']
        },
        'consumer_sentiment': {
            'series_id': 'UMCSENT',
            'name': 'Consumer Sentiment (U of Michigan)',
            'frequency': 'monthly',
            'impact': 'medium',
            'affects': ['equities']
        },
        'durable_goods': {
            'series_id': 'DGORDER',
            'name': 'Durable Goods Orders',
            'frequency': 'monthly',
            'impact': 'medium',
            'transform': 'pct_change',
            'affects': ['equities']
        },
        '10y_yield': {
            'series_id': 'DGS10',
            'name': '10-Year Treasury Yield',
            'frequency': 'daily',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates']
        },
        '2y_yield': {
            'series_id': 'DGS2',
            'name': '2-Year Treasury Yield',
            'frequency': 'daily',
            'impact': 'high',
            'affects': ['equities', 'fx', 'rates']
        },
        'yield_curve': {
            'series_id': 'T10Y2Y',
            'name': '10Y-2Y Treasury Spread',
            'frequency': 'daily',
            'impact': 'high',
            'affects': ['equities', 'rates']
        }
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache"):
        """Initialize the macro data fetcher.

        Args:
            api_key: FRED API key. If not provided, looks for FRED_API_KEY env var
            cache_dir: Directory for caching data
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY', '')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("No FRED API key provided. Using demo mode with sample data.")

    def _get_cache_path(self, series_id: str) -> Path:
        """Get cache file path for a series."""
        return self.cache_dir / f"{series_id}_data.pkl"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False

        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - modified_time
        return age < timedelta(hours=max_age_hours)

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch a FRED data series.

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with date index and 'value' column
        """
        cache_path = self._get_cache_path(series_id)

        # Check cache first
        if use_cache and self._is_cache_valid(cache_path):
            logger.debug(f"Loading {series_id} from cache")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*15)).strftime('%Y-%m-%d')

        # Use demo data if no API key
        if not self.api_key:
            return self._generate_demo_data(series_id, start_date, end_date)

        # Fetch from FRED API
        url = f"{self.FRED_BASE_URL}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            if not observations:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame()

            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date')[['value']].dropna()

            # Cache the data
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)

            logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching {series_id}: {e}")
            # Try to load from cache even if expired
            if cache_path.exists():
                logger.info(f"Using expired cache for {series_id}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            return self._generate_demo_data(series_id, start_date, end_date)

    def _generate_demo_data(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate realistic demo data for testing without API key."""
        logger.info(f"Generating demo data for {series_id}")

        dates = pd.date_range(start=start_date, end=end_date, freq='MS')

        # Generate realistic values based on series type
        np.random.seed(hash(series_id) % 2**32)

        if 'CPI' in series_id.upper():
            # CPI starts around 250 and grows ~2-3% annually
            base = 250
            trend = np.linspace(0, len(dates) * 0.002, len(dates))
            noise = np.random.randn(len(dates)) * 0.3
            values = base + base * (trend + np.cumsum(noise) * 0.001)

        elif 'PAYEMS' in series_id.upper():
            # Non-farm payrolls in thousands
            base = 150000
            trend = np.linspace(0, len(dates) * 100, len(dates))
            noise = np.random.randn(len(dates)) * 200
            values = base + trend + np.cumsum(noise)

        elif 'UNRATE' in series_id.upper():
            # Unemployment rate 3-10%
            base = 5.0
            cycle = 2 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
            noise = np.random.randn(len(dates)) * 0.3
            values = base + cycle + noise
            values = np.clip(values, 3, 10)

        elif 'FEDFUNDS' in series_id.upper() or 'DFED' in series_id.upper():
            # Fed funds rate 0-6%
            base = 3.0
            cycle = 2 * np.sin(np.linspace(0, 2*np.pi, len(dates)))
            values = base + cycle + np.random.randn(len(dates)) * 0.1
            values = np.clip(values, 0, 6)

        elif 'NAPM' in series_id.upper() or 'PMI' in series_id.upper():
            # PMI around 50
            values = 50 + np.random.randn(len(dates)) * 5

        elif 'GDP' in series_id.upper():
            # GDP in billions or growth rate
            if 'A191' in series_id:  # Growth rate
                values = 2 + np.random.randn(len(dates)) * 2
            else:
                base = 20000
                trend = np.linspace(0, len(dates) * 50, len(dates))
                values = base + trend + np.random.randn(len(dates)) * 100

        elif 'DGS' in series_id.upper() or 'T10Y' in series_id.upper():
            # Treasury yields
            base = 3.0
            cycle = np.sin(np.linspace(0, 4*np.pi, len(dates)))
            values = base + cycle + np.random.randn(len(dates)) * 0.2
            values = np.clip(values, 0.5, 6)

        elif 'UMCSENT' in series_id.upper():
            # Consumer sentiment 60-100
            values = 80 + np.random.randn(len(dates)) * 10
            values = np.clip(values, 60, 100)

        else:
            # Generic economic indicator
            base = 100
            trend = np.linspace(0, len(dates) * 0.1, len(dates))
            noise = np.random.randn(len(dates)) * 2
            values = base + trend + np.cumsum(noise) * 0.1

        df = pd.DataFrame({'value': values}, index=dates)
        return df

    def fetch_indicator(
        self,
        indicator_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch data for a named indicator.

        Args:
            indicator_key: Key from INDICATORS dict (e.g., 'cpi', 'nfp')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with indicator data
        """
        if indicator_key not in self.INDICATORS:
            logger.error(f"Unknown indicator: {indicator_key}")
            return pd.DataFrame()

        indicator = self.INDICATORS[indicator_key]
        df = self.fetch_series(indicator['series_id'], start_date, end_date)

        if df.empty:
            return df

        # Apply transformations if specified
        transform = indicator.get('transform')
        if transform == 'diff':
            df['value'] = df['value'].diff()
        elif transform == 'pct_change':
            df['value'] = df['value'].pct_change() * 100
        elif transform == 'yoy_pct':
            df['value'] = df['value'].pct_change(12) * 100

        df = df.dropna()
        df['indicator'] = indicator_key
        df['name'] = indicator['name']

        return df

    def fetch_all_indicators(
        self,
        indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple indicators.

        Args:
            indicators: List of indicator keys. If None, fetches all
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary of DataFrames keyed by indicator name
        """
        if indicators is None:
            indicators = list(self.INDICATORS.keys())

        results = {}
        for indicator_key in indicators:
            try:
                df = self.fetch_indicator(indicator_key, start_date, end_date)
                if not df.empty:
                    results[indicator_key] = df
            except Exception as e:
                logger.error(f"Error fetching {indicator_key}: {e}")

        return results

    def get_latest_values(
        self,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get the latest value for each indicator.

        Args:
            indicators: List of indicator keys

        Returns:
            DataFrame with latest values
        """
        data = self.fetch_all_indicators(indicators)

        latest = []
        for key, df in data.items():
            if not df.empty:
                latest_row = df.iloc[-1]
                latest.append({
                    'indicator': key,
                    'name': self.INDICATORS[key]['name'],
                    'value': latest_row['value'],
                    'date': df.index[-1],
                    'impact': self.INDICATORS[key]['impact']
                })

        return pd.DataFrame(latest)

    def get_indicator_info(self, indicator_key: str) -> Dict[str, Any]:
        """Get metadata about an indicator."""
        return self.INDICATORS.get(indicator_key, {})

    def calculate_release_surprise(
        self,
        indicator_key: str,
        actual: float,
        consensus: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate the surprise of a release vs consensus and historical.

        Args:
            indicator_key: The indicator key
            actual: The actual released value
            consensus: The consensus forecast (if available)

        Returns:
            Dictionary with surprise metrics
        """
        df = self.fetch_indicator(indicator_key)
        if df.empty:
            return {}

        # Calculate historical statistics
        recent = df['value'].tail(24)  # Last 2 years for monthly
        historical_mean = recent.mean()
        historical_std = recent.std()

        # Calculate z-score vs history
        z_score = (actual - historical_mean) / historical_std if historical_std > 0 else 0

        result = {
            'indicator': indicator_key,
            'actual': actual,
            'historical_mean': historical_mean,
            'historical_std': historical_std,
            'z_score_vs_history': z_score,
            'percentile': (recent < actual).mean() * 100
        }

        # Add consensus comparison if provided
        if consensus is not None:
            result['consensus'] = consensus
            result['surprise'] = actual - consensus
            result['surprise_pct'] = ((actual - consensus) / abs(consensus) * 100) if consensus != 0 else 0

        return result
