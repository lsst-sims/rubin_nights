import astropy.units as u
from astroplan import Observer
from astropy.time import Time, TimeDelta

__all__ = [
    "today_day_obs",
    "yesterday_day_obs",
    "time_to_day_obs",
    "day_obs_str_to_int",
    "day_obs_int_to_str",
    "day_obs_sunset_sunrise",
    "rtn045_plot_styles",
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
    day_obs = str(day_obs)
    day_obs = f"{day_obs[0:4]}-{day_obs[4:6]}-{day_obs[6:]}"
    return day_obs


def day_obs_str_to_int(day_obs: str) -> int:
    """Day_obs string YYYY-MM-DD to integer YYYYMMDD."""
    return int(day_obs.replace("-", ""))


def day_obs_sunset_sunrise(day_obs: str | int) -> tuple[Time, Time]:
    """Return the civil sunset and sunrise for day_obs.

    Parameters
    ----------
    day_obs : `str` or `int`
        Current day_obs in format YYYY-MM-DD or YYYYMMDD

    Returns
    -------
    sunset, sunrise : `Time`, `Time`
        The time of -6 degree (civil) sunset and sunrise.
        Science observations are generally expected from -12 degree twilight.
    """
    if isinstance(day_obs, int):
        day_obs = str(day_obs)
    if "-" not in day_obs:
        day_obs = day_obs_int_to_str(day_obs)
    day_obs_time = Time(f"{day_obs}T12:00:00", format="isot", scale="tai")
    observer = Observer.at_site("lsst")
    sunset = Time(observer.sun_set_time(day_obs_time, which="next", horizon=-6 * u.deg), format="jd")
    sunrise = Time(observer.sun_rise_time(day_obs_time, which="next", horizon=-6 * u.deg), format="jd")
    return (sunset, sunrise)


def rtn045_plot_styles() -> dict:
    plot_styles = {}
    plot_styles["band_colors_white"] = {
        "u": "#0c71ff",
        "g": "#49be61",
        "r": "#c61c00",
        "i": "#ffc200",
        "z": "#f341a2",
        "y": "#5d0000",
    }
    plot_styles["band_colors_black"] = {
        "u": "#3eb7ff",
        "g": "#30c39f",
        "r": "#ff7e00",
        "i": "#2af5ff",
        "z": "#a7f9c1",
        "y": "#fdc900",
    }
    plot_styles["band_symbols"] = {"u": "o", "g": "^", "r": "v", "i": "s", "z": "*", "y": "p"}
    plot_styles["band_linestyles"] = {
        "u": "--",
        "g": ":",
        "r": "-",
        "i": "-.",
        "z": (0, (3, 5, 1, 5, 1, 5)),
        "y": (0, (3, 1, 1, 1)),
    }
    return plot_styles
