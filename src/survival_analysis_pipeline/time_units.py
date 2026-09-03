"""The dataset's time unit, declared once and threaded everywhere.

Durations, horizons, and predictions all live in one unit the user declares
(--time-unit, days by default). The model itself never needs to know it: an
AFT model on log time and a concordance index are both scale-free. The unit
matters in exactly two places. Label re-censoring compares durations against
calendar spans, so those spans must be converted into the dataset's unit, and
the report and figures must name the unit rather than assuming days.

Months and years have no fixed calendar length, so their conversions use
fixed averages (365.25 days per year, a twelfth of that per month). The
error against any real month or year is under two percent, which is
negligible next to the day-level resolution of a start-date column.

The abbreviations label horizons in file names, metrics keys, and output
columns ("90d", "p_survive_24h"). "d" for days keeps every existing
day-based artifact name and metrics key exactly as it was.
"""

from __future__ import annotations

_DAY_SECONDS = 86400.0

# unit -> (singular form for prose, horizon abbreviation, seconds per unit)
_UNITS: dict[str, tuple[str, str, float]] = {
    "seconds": ("second", "s", 1.0),
    "minutes": ("minute", "min", 60.0),
    "hours": ("hour", "h", 3600.0),
    "days": ("day", "d", _DAY_SECONDS),
    "weeks": ("week", "w", 7 * _DAY_SECONDS),
    "months": ("month", "mo", 365.25 / 12 * _DAY_SECONDS),
    "years": ("year", "y", 365.25 * _DAY_SECONDS),
}

# In size order, for CLI choices and error messages.
TIME_UNITS: tuple[str, ...] = tuple(_UNITS)


def check_time_unit(unit: str) -> str:
    """The unit if it is supported, else a ValueError naming the options."""
    if unit not in _UNITS:
        raise ValueError(f"unknown time unit {unit!r}; expected one of {', '.join(TIME_UNITS)}")
    return unit


def unit_seconds(unit: str) -> float:
    """Seconds in one timestep of the unit."""
    return _UNITS[check_time_unit(unit)][2]


def unit_singular(unit: str) -> str:
    """The unit's singular name for prose."""
    return _UNITS[check_time_unit(unit)][0]


def unit_abbrev(unit: str) -> str:
    """The unit's abbreviation for column keys and axis labels."""
    return _UNITS[check_time_unit(unit)][1]


def horizon_label(h: float) -> str:
    """A horizon's spelling in keys, file names, and prose: 90.0 -> '90',
    0.25 -> '0.25'. Whole numbers drop the decimal so day-based artifacts
    keep their exact historical names; fractional horizons, which int()
    used to collapse into colliding keys, stay distinct."""
    hf = float(h)
    return str(int(hf)) if hf.is_integer() else f"{hf:g}"
