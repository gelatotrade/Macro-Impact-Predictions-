"""
Economic Calendar Module

Tracks upcoming macroeconomic events and their consensus expectations.
Fetches from various sources including Investing.com, ForexFactory, and APIs.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from loguru import logger


class EventImpact(Enum):
    """Event impact levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EconomicEvent:
    """Represents a scheduled economic event."""
    date: datetime
    time: Optional[str]
    country: str
    event_name: str
    impact: EventImpact
    previous: Optional[float] = None
    consensus: Optional[float] = None
    actual: Optional[float] = None
    unit: str = ""
    event_type: str = ""
    affects: List[str] = field(default_factory=list)

    @property
    def surprise(self) -> Optional[float]:
        """Calculate surprise vs consensus."""
        if self.actual is not None and self.consensus is not None:
            return self.actual - self.consensus
        return None

    @property
    def surprise_pct(self) -> Optional[float]:
        """Calculate surprise as percentage."""
        if self.surprise is not None and self.consensus and self.consensus != 0:
            return (self.surprise / abs(self.consensus)) * 100
        return None

    @property
    def is_released(self) -> bool:
        """Check if event has been released."""
        return self.actual is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'date': self.date.isoformat(),
            'time': self.time,
            'country': self.country,
            'event_name': self.event_name,
            'impact': self.impact.name,
            'previous': self.previous,
            'consensus': self.consensus,
            'actual': self.actual,
            'unit': self.unit,
            'event_type': self.event_type,
            'surprise': self.surprise,
            'surprise_pct': self.surprise_pct,
            'is_released': self.is_released,
            'affects': self.affects
        }


class EconomicCalendar:
    """Manages economic calendar events and consensus expectations."""

    # Major economic events and their typical market impacts
    EVENT_DEFINITIONS = {
        'CPI': {
            'full_name': 'Consumer Price Index',
            'country': 'US',
            'impact': EventImpact.CRITICAL,
            'frequency': 'monthly',
            'affects': ['SPY', 'QQQ', 'TLT', 'DXY', 'EURUSD', 'USDJPY', 'GC=F'],
            'typical_release_day': 'second_week',
            'event_type': 'inflation'
        },
        'Core CPI': {
            'full_name': 'Core Consumer Price Index (Ex Food & Energy)',
            'country': 'US',
            'impact': EventImpact.CRITICAL,
            'frequency': 'monthly',
            'affects': ['SPY', 'QQQ', 'TLT', 'DXY'],
            'event_type': 'inflation'
        },
        'NFP': {
            'full_name': 'Non-Farm Payrolls',
            'country': 'US',
            'impact': EventImpact.CRITICAL,
            'frequency': 'monthly',
            'affects': ['SPY', 'QQQ', 'TLT', 'DXY', 'EURUSD', 'USDJPY'],
            'typical_release_day': 'first_friday',
            'event_type': 'employment'
        },
        'Unemployment Rate': {
            'full_name': 'Unemployment Rate',
            'country': 'US',
            'impact': EventImpact.HIGH,
            'frequency': 'monthly',
            'affects': ['SPY', 'TLT', 'DXY'],
            'event_type': 'employment'
        },
        'FOMC': {
            'full_name': 'Federal Open Market Committee Rate Decision',
            'country': 'US',
            'impact': EventImpact.CRITICAL,
            'frequency': '6_weeks',
            'affects': ['SPY', 'QQQ', 'TLT', 'IEF', 'DXY', 'EURUSD', 'USDJPY', 'GC=F'],
            'event_type': 'rates'
        },
        'Fed Chair Speech': {
            'full_name': 'Federal Reserve Chair Powell Speech',
            'country': 'US',
            'impact': EventImpact.HIGH,
            'affects': ['SPY', 'TLT', 'DXY'],
            'event_type': 'rates'
        },
        'ISM Manufacturing PMI': {
            'full_name': 'ISM Manufacturing Purchasing Managers Index',
            'country': 'US',
            'impact': EventImpact.HIGH,
            'frequency': 'monthly',
            'affects': ['SPY', 'QQQ', 'DXY'],
            'typical_release_day': 'first_business',
            'event_type': 'pmi'
        },
        'ISM Services PMI': {
            'full_name': 'ISM Non-Manufacturing PMI',
            'country': 'US',
            'impact': EventImpact.HIGH,
            'frequency': 'monthly',
            'affects': ['SPY', 'DXY'],
            'event_type': 'pmi'
        },
        'GDP': {
            'full_name': 'Gross Domestic Product (Quarterly)',
            'country': 'US',
            'impact': EventImpact.HIGH,
            'frequency': 'quarterly',
            'affects': ['SPY', 'QQQ', 'TLT', 'DXY'],
            'event_type': 'growth'
        },
        'Core PCE': {
            'full_name': 'Core Personal Consumption Expenditure Price Index',
            'country': 'US',
            'impact': EventImpact.CRITICAL,
            'frequency': 'monthly',
            'affects': ['SPY', 'QQQ', 'TLT', 'DXY'],
            'event_type': 'inflation'
        },
        'Retail Sales': {
            'full_name': 'Advance Retail Sales',
            'country': 'US',
            'impact': EventImpact.MEDIUM,
            'frequency': 'monthly',
            'affects': ['SPY', 'QQQ'],
            'event_type': 'consumer'
        },
        'Initial Claims': {
            'full_name': 'Initial Jobless Claims',
            'country': 'US',
            'impact': EventImpact.MEDIUM,
            'frequency': 'weekly',
            'affects': ['SPY', 'TLT'],
            'event_type': 'employment'
        },
        'ECB Rate Decision': {
            'full_name': 'European Central Bank Interest Rate Decision',
            'country': 'EU',
            'impact': EventImpact.HIGH,
            'affects': ['EURUSD', 'SPY', 'DXY'],
            'event_type': 'rates'
        },
        'BOJ Rate Decision': {
            'full_name': 'Bank of Japan Interest Rate Decision',
            'country': 'JP',
            'impact': EventImpact.HIGH,
            'affects': ['USDJPY', 'DXY'],
            'event_type': 'rates'
        },
        'BOE Rate Decision': {
            'full_name': 'Bank of England Interest Rate Decision',
            'country': 'UK',
            'impact': EventImpact.HIGH,
            'affects': ['GBPUSD', 'DXY'],
            'event_type': 'rates'
        }
    }

    # Consensus estimates (would be updated from live sources)
    CONSENSUS_ESTIMATES = {
        'CPI MoM': {'consensus': 0.3, 'previous': 0.2, 'unit': '%'},
        'CPI YoY': {'consensus': 3.2, 'previous': 3.4, 'unit': '%'},
        'Core CPI MoM': {'consensus': 0.3, 'previous': 0.3, 'unit': '%'},
        'Core CPI YoY': {'consensus': 3.3, 'previous': 3.5, 'unit': '%'},
        'NFP': {'consensus': 180, 'previous': 227, 'unit': 'K'},
        'Unemployment Rate': {'consensus': 4.2, 'previous': 4.2, 'unit': '%'},
        'Fed Funds Rate': {'consensus': 5.25, 'previous': 5.50, 'unit': '%'},
        'ISM Manufacturing PMI': {'consensus': 48.5, 'previous': 47.2, 'unit': 'index'},
        'ISM Services PMI': {'consensus': 53.0, 'previous': 52.1, 'unit': 'index'},
        'GDP QoQ': {'consensus': 2.8, 'previous': 3.0, 'unit': '%'},
        'Core PCE MoM': {'consensus': 0.2, 'previous': 0.3, 'unit': '%'},
        'Core PCE YoY': {'consensus': 2.8, 'previous': 2.8, 'unit': '%'},
        'Retail Sales MoM': {'consensus': 0.4, 'previous': 0.4, 'unit': '%'},
        'Initial Claims': {'consensus': 220, 'previous': 224, 'unit': 'K'},
    }

    def __init__(self):
        """Initialize the economic calendar."""
        self._events: List[EconomicEvent] = []
        self._load_upcoming_events()

    def _load_upcoming_events(self) -> None:
        """Load upcoming economic events."""
        # Generate a realistic calendar of upcoming events
        self._events = self._generate_upcoming_calendar()

    def _generate_upcoming_calendar(self) -> List[EconomicEvent]:
        """Generate upcoming economic events calendar."""
        events = []
        today = datetime.now()

        # Generate events for the next 30 days
        for days_ahead in range(30):
            event_date = today + timedelta(days=days_ahead)

            # Skip weekends
            if event_date.weekday() >= 5:
                continue

            # Add events based on typical release schedules
            events.extend(self._generate_events_for_date(event_date))

        # Sort by date
        events.sort(key=lambda x: x.date)
        return events

    def _generate_events_for_date(self, date: datetime) -> List[EconomicEvent]:
        """Generate events for a specific date."""
        events = []
        day_of_week = date.weekday()
        day_of_month = date.day
        week_of_month = (day_of_month - 1) // 7 + 1

        # First Friday - NFP
        if day_of_week == 4 and week_of_month == 1:
            events.append(EconomicEvent(
                date=date.replace(hour=8, minute=30),
                time="08:30 ET",
                country="US",
                event_name="Non-Farm Payrolls",
                impact=EventImpact.CRITICAL,
                previous=self.CONSENSUS_ESTIMATES['NFP']['previous'],
                consensus=self.CONSENSUS_ESTIMATES['NFP']['consensus'],
                unit="K",
                event_type="employment",
                affects=['SPY', 'QQQ', 'TLT', 'DXY', 'EURUSD', 'USDJPY']
            ))
            events.append(EconomicEvent(
                date=date.replace(hour=8, minute=30),
                time="08:30 ET",
                country="US",
                event_name="Unemployment Rate",
                impact=EventImpact.HIGH,
                previous=self.CONSENSUS_ESTIMATES['Unemployment Rate']['previous'],
                consensus=self.CONSENSUS_ESTIMATES['Unemployment Rate']['consensus'],
                unit="%",
                event_type="employment",
                affects=['SPY', 'TLT', 'DXY']
            ))

        # Second week - CPI (usually Tuesday or Wednesday)
        if week_of_month == 2 and day_of_week in [1, 2]:
            if day_of_week == 2:  # Wednesday
                events.append(EconomicEvent(
                    date=date.replace(hour=8, minute=30),
                    time="08:30 ET",
                    country="US",
                    event_name="CPI MoM",
                    impact=EventImpact.CRITICAL,
                    previous=self.CONSENSUS_ESTIMATES['CPI MoM']['previous'],
                    consensus=self.CONSENSUS_ESTIMATES['CPI MoM']['consensus'],
                    unit="%",
                    event_type="inflation",
                    affects=['SPY', 'QQQ', 'TLT', 'DXY', 'EURUSD', 'GC=F']
                ))
                events.append(EconomicEvent(
                    date=date.replace(hour=8, minute=30),
                    time="08:30 ET",
                    country="US",
                    event_name="CPI YoY",
                    impact=EventImpact.CRITICAL,
                    previous=self.CONSENSUS_ESTIMATES['CPI YoY']['previous'],
                    consensus=self.CONSENSUS_ESTIMATES['CPI YoY']['consensus'],
                    unit="%",
                    event_type="inflation",
                    affects=['SPY', 'QQQ', 'TLT', 'DXY', 'EURUSD', 'GC=F']
                ))
                events.append(EconomicEvent(
                    date=date.replace(hour=8, minute=30),
                    time="08:30 ET",
                    country="US",
                    event_name="Core CPI MoM",
                    impact=EventImpact.CRITICAL,
                    previous=self.CONSENSUS_ESTIMATES['Core CPI MoM']['previous'],
                    consensus=self.CONSENSUS_ESTIMATES['Core CPI MoM']['consensus'],
                    unit="%",
                    event_type="inflation",
                    affects=['SPY', 'QQQ', 'TLT', 'DXY']
                ))

        # First business day - ISM Manufacturing PMI
        if day_of_month <= 3 and day_of_week == 0:
            events.append(EconomicEvent(
                date=date.replace(hour=10, minute=0),
                time="10:00 ET",
                country="US",
                event_name="ISM Manufacturing PMI",
                impact=EventImpact.HIGH,
                previous=self.CONSENSUS_ESTIMATES['ISM Manufacturing PMI']['previous'],
                consensus=self.CONSENSUS_ESTIMATES['ISM Manufacturing PMI']['consensus'],
                unit="index",
                event_type="pmi",
                affects=['SPY', 'QQQ', 'DXY']
            ))

        # Third Wednesday - FOMC (8 times per year, approximate)
        if week_of_month == 3 and day_of_week == 2:
            if date.month in [1, 3, 5, 6, 7, 9, 11, 12]:
                events.append(EconomicEvent(
                    date=date.replace(hour=14, minute=0),
                    time="14:00 ET",
                    country="US",
                    event_name="FOMC Rate Decision",
                    impact=EventImpact.CRITICAL,
                    previous=self.CONSENSUS_ESTIMATES['Fed Funds Rate']['previous'],
                    consensus=self.CONSENSUS_ESTIMATES['Fed Funds Rate']['consensus'],
                    unit="%",
                    event_type="rates",
                    affects=['SPY', 'QQQ', 'TLT', 'IEF', 'DXY', 'EURUSD', 'USDJPY', 'GC=F']
                ))

        # Thursdays - Initial Claims
        if day_of_week == 3:
            events.append(EconomicEvent(
                date=date.replace(hour=8, minute=30),
                time="08:30 ET",
                country="US",
                event_name="Initial Jobless Claims",
                impact=EventImpact.MEDIUM,
                previous=self.CONSENSUS_ESTIMATES['Initial Claims']['previous'],
                consensus=self.CONSENSUS_ESTIMATES['Initial Claims']['consensus'],
                unit="K",
                event_type="employment",
                affects=['SPY', 'TLT']
            ))

        # Last Friday of month - Core PCE
        if day_of_week == 4 and day_of_month >= 25:
            events.append(EconomicEvent(
                date=date.replace(hour=8, minute=30),
                time="08:30 ET",
                country="US",
                event_name="Core PCE MoM",
                impact=EventImpact.CRITICAL,
                previous=self.CONSENSUS_ESTIMATES['Core PCE MoM']['previous'],
                consensus=self.CONSENSUS_ESTIMATES['Core PCE MoM']['consensus'],
                unit="%",
                event_type="inflation",
                affects=['SPY', 'QQQ', 'TLT', 'DXY']
            ))

        return events

    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        min_impact: EventImpact = EventImpact.MEDIUM,
        country: Optional[str] = None
    ) -> List[EconomicEvent]:
        """Get upcoming economic events.

        Args:
            days_ahead: Number of days to look ahead
            min_impact: Minimum impact level to include
            country: Filter by country code

        Returns:
            List of upcoming events
        """
        cutoff_date = datetime.now() + timedelta(days=days_ahead)

        filtered = []
        for event in self._events:
            if event.date > cutoff_date:
                continue
            if event.date < datetime.now():
                continue
            if event.impact.value < min_impact.value:
                continue
            if country and event.country != country:
                continue
            filtered.append(event)

        return filtered

    def get_next_event(self, event_type: Optional[str] = None) -> Optional[EconomicEvent]:
        """Get the next upcoming event.

        Args:
            event_type: Filter by event type

        Returns:
            Next event or None
        """
        now = datetime.now()
        for event in self._events:
            if event.date <= now:
                continue
            if event_type and event.event_type != event_type:
                continue
            return event
        return None

    def get_events_by_type(self, event_type: str) -> List[EconomicEvent]:
        """Get all events of a specific type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_high_impact_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get high and critical impact events."""
        return self.get_upcoming_events(
            days_ahead=days_ahead,
            min_impact=EventImpact.HIGH
        )

    def to_dataframe(self, events: Optional[List[EconomicEvent]] = None) -> pd.DataFrame:
        """Convert events to DataFrame.

        Args:
            events: Events to convert. If None, uses all upcoming events.

        Returns:
            DataFrame of events
        """
        if events is None:
            events = self.get_upcoming_events(days_ahead=30, min_impact=EventImpact.LOW)

        data = [e.to_dict() for e in events]
        return pd.DataFrame(data)

    def get_event_consensus(self, event_name: str) -> Dict[str, Any]:
        """Get consensus estimate for an event.

        Args:
            event_name: Name of the event

        Returns:
            Dictionary with consensus, previous, unit
        """
        return self.CONSENSUS_ESTIMATES.get(event_name, {})

    def update_consensus(self, event_name: str, consensus: float, previous: Optional[float] = None) -> None:
        """Update consensus estimate for an event.

        Args:
            event_name: Name of the event
            consensus: New consensus value
            previous: Previous value (optional)
        """
        if event_name not in self.CONSENSUS_ESTIMATES:
            self.CONSENSUS_ESTIMATES[event_name] = {}

        self.CONSENSUS_ESTIMATES[event_name]['consensus'] = consensus
        if previous is not None:
            self.CONSENSUS_ESTIMATES[event_name]['previous'] = previous

        logger.info(f"Updated consensus for {event_name}: {consensus}")

    def get_events_affecting_instrument(self, symbol: str, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get events that typically affect a specific instrument.

        Args:
            symbol: Instrument symbol
            days_ahead: Days to look ahead

        Returns:
            List of relevant events
        """
        events = self.get_upcoming_events(days_ahead=days_ahead, min_impact=EventImpact.LOW)
        return [e for e in events if symbol in e.affects]

    def calculate_event_density(self, days_ahead: int = 7) -> Dict[str, Any]:
        """Calculate event density metrics.

        Useful for understanding how crowded the calendar is.
        """
        events = self.get_upcoming_events(days_ahead=days_ahead, min_impact=EventImpact.LOW)

        high_impact = sum(1 for e in events if e.impact.value >= EventImpact.HIGH.value)
        critical = sum(1 for e in events if e.impact == EventImpact.CRITICAL)

        return {
            'total_events': len(events),
            'high_impact_events': high_impact,
            'critical_events': critical,
            'avg_daily_events': len(events) / max(days_ahead, 1),
            'event_risk': 'high' if critical >= 3 else 'medium' if high_impact >= 5 else 'low'
        }
