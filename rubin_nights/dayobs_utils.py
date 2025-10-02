import datetime

import astropy.units as u
from astroplan import Observer
from astropy.coordinates.errors import UnknownSiteException
from astropy.time import Time, TimeDelta

__all__ = [
    "today_day_obs",
    "yesterday_day_obs",
    "time_to_day_obs",
    "day_obs_str_to_int",
    "day_obs_int_to_str",
    "day_obs_to_date",
    "day_obs_to_time",
    "day_obs_sunset_sunrise",
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
    if isinstance(day_obs, int):
        day_obs_str = str(day_obs)
    else:
        day_obs_str = day_obs
    if "-" not in day_obs_str:
        day_obs_str = day_obs_int_to_str(int(day_obs))
    day_obs_time = Time(f"{day_obs_str}T12:00:00", format="isot", scale="tai")
    try:
        observer = Observer.at_site("Rubin")
    except UnknownSiteException:
        # Better to use Rubin, but old astropy installs might not have it.
        observer = Observer.at_site("Cerro Pachon")
    sunset = Time(
        observer.sun_set_time(day_obs_time, which="next", horizon=sun_alt * u.deg), format="jd", scale="tai"
    )
    sunrise = Time(
        observer.sun_rise_time(day_obs_time, which="next", horizon=sun_alt * u.deg), format="jd", scale="tai"
    )
    return (sunset, sunrise)
