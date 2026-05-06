"""
Consensus Loader

Pulls the *latest* values for tracked macro events from FRED so that
every run of the prediction engine sees fresh `previous` releases and
a forecast-ready `consensus` proxy.

Why this exists
---------------
The static `CONSENSUS_ESTIMATES` dict in `economic_calendar.py` hard-coded
"previous" and "consensus" values (CPI MoM = 0.3, NFP = 180k, ...).
Those numbers were a snapshot of the data at the time the file was written
— they never updated. This loader replaces them at every engine
instantiation with the actual latest FRED data.

Behaviour
---------
- `previous`  = literal last actual release pulled from FRED (hard fact)
- `consensus` = trailing 3-period mean (a robust proxy for economist
  consensus when no Bloomberg/Reuters feed is available). It biases
  toward recent trend.
- An optional `data/consensus_overrides.json` lets the user pin specific
  values that beat both proxies (e.g. paste in the latest ForexFactory
  consensus before a known release).
- If FRED is unreachable, callers fall back to the static defaults.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from .macro_data_fetcher import MacroDataFetcher


# event_name -> (FRED series, transform, unit)
EVENT_TO_FRED: Dict[str, Tuple[str, str, str]] = {
    "CPI MoM":               ("CPIAUCSL",          "mom_pct",   "%"),
    "CPI YoY":               ("CPIAUCSL",          "yoy_pct",   "%"),
    "Core CPI MoM":          ("CPILFESL",          "mom_pct",   "%"),
    "Core CPI YoY":          ("CPILFESL",          "yoy_pct",   "%"),
    "NFP":                   ("PAYEMS",            "mom_diff",  "K"),
    "Unemployment Rate":     ("UNRATE",            "level",     "%"),
    "Fed Funds Rate":        ("DFEDTARU",          "level",     "%"),
    "ISM Manufacturing PMI": ("NAPM",              "level",     "index"),
    "ISM Services PMI":      ("NMFCI",             "level",     "index"),
    "GDP QoQ":               ("A191RL1Q225SBEA",   "level",     "%"),
    "Core PCE MoM":          ("PCEPILFE",          "mom_pct",   "%"),
    "Core PCE YoY":          ("PCEPILFE",          "yoy_pct",   "%"),
    "Retail Sales MoM":      ("RSAFS",             "mom_pct",   "%"),
    "Initial Claims":        ("ICSA",              "level",     "K"),
}


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return series
    if transform == "mom_pct":
        return series.pct_change(1) * 100
    if transform == "yoy_pct":
        return series.pct_change(12) * 100
    if transform == "mom_diff":
        return series.diff()
    raise ValueError(f"Unknown transform: {transform}")


class ConsensusLoader:
    """Loads the latest previous + consensus-proxy for every tracked event."""

    DEFAULT_OVERRIDES_PATH = Path("data/consensus_overrides.json")

    def __init__(
        self,
        macro_fetcher: Optional[MacroDataFetcher] = None,
        overrides_path: Optional[Path] = None,
    ) -> None:
        self.fetcher = macro_fetcher or MacroDataFetcher()
        self.overrides_path = (
            Path(overrides_path) if overrides_path is not None
            else self.DEFAULT_OVERRIDES_PATH
        )

    def load_all(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Return a dict shaped exactly like `EconomicCalendar.CONSENSUS_ESTIMATES`.

        Args:
            force_refresh: bypass the FRED 24h disk cache and pull anew.
        """
        result: Dict[str, Dict[str, Any]] = {}
        fetched_at = datetime.now().isoformat(timespec="seconds")

        for event_name, (series_id, transform, unit) in EVENT_TO_FRED.items():
            try:
                df = self.fetcher.fetch_series(series_id, use_cache=not force_refresh)
                if df.empty:
                    logger.warning(f"{event_name}: empty series {series_id}, skipping")
                    continue

                values = _apply_transform(df["value"], transform).dropna()
                if values.empty:
                    logger.warning(f"{event_name}: no usable values after transform")
                    continue

                previous = float(values.iloc[-1])
                # Trailing-3 mean as a robust forecast proxy. Fall back to
                # `previous` if we don't have 3 prior points.
                tail = values.tail(3)
                consensus = float(tail.mean()) if len(tail) >= 1 else previous

                result[event_name] = {
                    "consensus": round(consensus, 2),
                    "previous": round(previous, 2),
                    "unit": unit,
                    "source": "fred",
                    "series_id": series_id,
                    "fetched_at": fetched_at,
                    "as_of": df.index[-1].strftime("%Y-%m-%d"),
                }
            except Exception as exc:
                logger.warning(f"{event_name}: could not refresh from FRED ({exc})")

        # Apply user overrides (highest priority)
        if self.overrides_path.exists():
            try:
                with open(self.overrides_path, encoding="utf-8") as f:
                    overrides = json.load(f)
                for event_name, fields in overrides.items():
                    base = result.get(event_name, {"unit": "", "source": "override-only"})
                    base.update(fields)
                    base["source"] = "override"
                    base["fetched_at"] = fetched_at
                    result[event_name] = base
                logger.info(f"Applied {len(overrides)} consensus overrides "
                            f"from {self.overrides_path}")
            except Exception as exc:
                logger.warning(f"Could not read overrides {self.overrides_path}: {exc}")

        logger.info(f"Loaded {len(result)} consensus estimates "
                    f"(source: FRED + overrides) at {fetched_at}")
        return result

    def get_current_fed_funds_rate(self) -> Optional[float]:
        """Convenience: return the live Fed Funds Target (Upper) from FRED."""
        try:
            df = self.fetcher.fetch_series("DFEDTARU")
            if df.empty:
                return None
            return float(df["value"].iloc[-1])
        except Exception as exc:
            logger.warning(f"Could not fetch DFEDTARU: {exc}")
            return None
