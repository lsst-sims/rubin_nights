import datetime
import functools

import astropy.units as u
import numpy as np
from astroplan import Observer
from astropy.coordinates.errors import UnknownSiteException
from astropy.time import Time, TimeDelta

__all__ = [
    "today_day_obs",
    "yesterday_day_obs",
    "time_to_day_obs",
    "time_to_day_obs_int",
    "day_obs_str_to_int",
    "day_obs_int_to_str",
    "day_obs_to_date",
    "day_obs_to_time",
    "rubin_observer",
    "day_obs_sunset_sunrise",
    "estimated_baseline_visit_range",
]


def today_day_obs() -> str:
    """Return the day_obs for today, formatted as YYYY-MM-DD."""
    return time_to_day_obs(Time.now())


def yesterday_day_obs() -> str:
    """Return the day_obs for yesterday, formatted as YYYY-MM-DD."""
    return time_to_day_obs(Time.now() - TimeDelta(1, format="jd"))


def time_to_day_obs(time: Time) -> str:
    """Return day_obs for astropy Time, formatted as YYYY-MM-DD."""
    return Time(int(time.mjd - 0.5), format="mjd", scale="utc").iso[0:10]


def time_to_day_obs_int(time: Time) -> int:
    """Return day_obs for astropy Time, integer YYYYMMDD."""
    day_obs_str = Time(int(time.mjd - 0.5), format="mjd", scale="utc").iso[0:10]
    return day_obs_str_to_int(day_obs_str)


def day_obs_int_to_str(day_obs: int) -> str:
    """Day_obs integer YYYYMMDD transformed to string YYYY-MM-DD."""
    day_obs_str = str(day_obs)
    return f"{day_obs_str[0:4]}-{day_obs_str[4:6]}-{day_obs_str[6:]}"


def day_obs_str_to_int(day_obs: str) -> int:
    """Day_obs string YYYY-MM-DD to integer YYYYMMDD."""
    return int(day_obs.replace("-", ""))


def day_obs_to_date(day_obs: int | str) -> datetime.date:
    day_obs_str = str(day_obs)
    if "-" not in day_obs_str:
        day_obs_str = day_obs_int_to_str(int(day_obs))
    vals = day_obs_str.split("-")
    return datetime.date(int(vals[0]), int(vals[1]), int(vals[2]))


def day_obs_to_time(day_obs: int | str) -> Time:
    """Day_obs int or string to astropy Time."""
    try:
        day_obs_int = int(day_obs)
        return Time(f"{day_obs_int_to_str(day_obs_int)}T12:00:00", format="isot", scale="tai")
    except ValueError:
        return Time(f"{day_obs}T12:00:00", format="isot", scale="tai")


@functools.cache
def rubin_observer() -> Observer:
    try:
        observer = Observer.at_site("Rubin")
    except UnknownSiteException:
        # Better to use Rubin, but old astropy installs might not have it.
        observer = Observer.at_site("Cerro Pachon")
    return observer


def day_obs_sunset_sunrise(day_obs: str | int, sun_alt: float = -12) -> tuple[Time, Time]:
    """Return the civil sunset and sunrise for day_obs.

    Parameters
    ----------
    day_obs
        Current day_obs in format YYYY-MM-DD or YYYYMMDD
    sun_alt
        Altitude (in degrees) of the sun at 'sunrise' and 'sunset'.

    Returns
    -------
    sunset, sunrise : `Time`, `Time`
        The time of -6 degree (civil) sunset and sunrise.
        Science observations are generally expected from -12 degree twilight.
    """
    if isinstance(day_obs, str):
        day_obs_str = day_obs
    else:
        day_obs_str = str(day_obs)
    # Sometimes we may be passed YYYYMMDD in a string already
    if "-" not in day_obs_str:
        day_obs_str = day_obs_int_to_str(int(day_obs))
    day_obs_time = Time(f"{day_obs_str}T12:00:00", format="isot", scale="tai")
    observer = rubin_observer()
    sunset = Time(
        observer.sun_set_time(day_obs_time, which="next", horizon=sun_alt * u.deg), format="jd", scale="tai"
    )
    sunrise = Time(
        observer.sun_rise_time(day_obs_time, which="next", horizon=sun_alt * u.deg), format="jd", scale="tai"
    )
    return (sunset, sunrise)


def estimated_baseline_visit_range(day_obs: int, relative_performance: float=1.0) -> dict[str, int]:
    """Estimate an average and likely upper limit for the number of visits on
    a given day_obs.

    Parameters
    ----------
    day_obs
        The day of interest.
    relative_performance
        The expected open shutter fraction ratio compared to the baseline
        value used in the simulations.
        The simulations used to estimate open shutter fraction here are v5.1.

    Returns
    -------
    estimate_visits : `dict` [`str` : `int`]
        A dictionary with `n_vis_ave` and `n_vis_high` keys, containing
        a range of estimated nvisits values. These estimates are based on
        open shutter fraction and night length.
    """
    sunset, sunrise = day_obs_sunset_sunrise(day_obs, sun_alt=-12)
    night_length = (sunrise - sunset).jd * 24 * 60 * 60  # seconds
    night = day_obs_to_time(day_obs)
    night_min = Time("2025-12-21T12:00:00")
    # The mean open shutter fraction in v5.1 runs is ~0.74
    # but this doesn't match average nvisits without modulation
    open_shutter_fraction = 0.74 - 0.02 * np.cos((night - night_min).jd / 365 * 2 * np.pi)
    estimate_nvis_ave = open_shutter_fraction * night_length / 30  # assume 30s visits
    estimate_nvis_ave = int(np.floor(estimate_nvis_ave * relative_performance))
    # The upper envelope of nvisits/night over many simulations + years
    # matches more closely with 0.78
    open_shutter_fraction = 0.78
    estimate_nvis_hi = open_shutter_fraction * night_length / 30  # assume 30s visits
    estimate_nvis_hi = int(np.floor(estimate_nvis_hi * relative_performance))
    return {"n_vis_ave": estimate_nvis_ave, "n_vis_high": estimate_nvis_hi}
