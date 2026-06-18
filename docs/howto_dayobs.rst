.. py:currentmodule:: rubin_nights

.. _howto_dayobs:

Day Obs Utilities
=================

``day_obs`` is the local (Chilean) calendar date of the start of an observing night —
the date of sunset.  Because the night spans midnight, a single ``day_obs`` value
covers approximately noon-to-noon UTC.  The :mod:`~rubin_nights.dayobs_utils` module
provides conversions between the two representations used in Rubin systems
(``YYYYMMDD`` integer and ``YYYY-MM-DD`` string), as well as conversions to and from
:class:`astropy.time.Time` and utilities for working with ranges of nights.

Additional utilities related to day_obs are available in:

* `schedview.DayObs <https://github.com/lsst/schedview/blob/main/schedview/dayobs.py>`_
* `lsst.summit_utils.dateTime <https://github.com/lsst-so/summit_utils/blob/main/python/lsst/summit/utils/dateTime.py>`_



Getting today's day_obs
------------------------

.. code-block:: python

    import rubin_nights.dayobs_utils as rn_dayobs

    today   = rn_dayobs.today_day_obs()      # "2026-06-05"
    yesterday = rn_dayobs.yesterday_day_obs() # "2026-06-04"
    tomorrow  = rn_dayobs.tomorrow_day_obs()  # "2026-06-06"

All three return a string in ``YYYY-MM-DD`` format.

Converting between formats
---------------------------

.. code-block:: python

    # int <-> string
    rn_dayobs.day_obs_int_to_str(20260605)  # "2026-06-05"
    rn_dayobs.day_obs_str_to_int("2026-06-05")  # 20260605

    # astropy Time -> day_obs
    from astropy.time import Time
    rn_dayobs.time_to_day_obs(Time.now())      # "2026-06-05" (str)
    rn_dayobs.time_to_day_obs_int(Time.now())  # 20260605 (int)

    # MJD -> day_obs integer (useful as a DataFrame column operation)
    rn_dayobs.mjd_to_dayobs(60831.5)  # 20260605

    # day_obs -> start of night as astropy Time (noon TAI)
    t_start = rn_dayobs.day_obs_to_time(20260605)
    t_start = rn_dayobs.day_obs_to_time("2026-06-05")  # either format accepted

    # day_obs -> Python date object
    rn_dayobs.day_obs_to_date(20260605)

Converting a DataFrame column of MJD values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~dayobs_utils.mjd_to_dayobs_array` converts MJD (in TAI) into day_obs integers.
This is convenient when working with opsim outputs which do not have a day_obs value.

.. code-block:: python

    visits["day_obs"] = rn_dayobs.mjd_to_dayobs_array(visits["observationStartMJD"])


Converting an EFD DataFrame time index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~influx_query.day_obs_from_efd_index_array` converts the EFD index times
into day_obs integers. This is convenient to group EFD results per day_obs.

.. code-block:: python

    from rubin_nights.influx_query import day_obs_from_efd_index_array
    efd_data["day_obs"] = day_obs_from_efd_index_array(efd_data.index)


Setting a time window for a night
-----------------------------------

The standard pattern for querying a full ``day_obs`` is a 24-hour window starting at
noon UTC.  :func:`~dayobs_utils.day_obs_to_time` returns that start time directly:

.. code-block:: python

    from astropy.time import TimeDelta

    day_obs = 20260605
    t_start = rn_dayobs.day_obs_to_time(day_obs)
    t_end = t_start + TimeDelta(1, format="jd")

Sunset and sunrise times
-------------------------

:func:`~dayobs_utils.day_obs_sunset_sunrise` returns the actual sunset and sunrise
times for a given ``day_obs`` at the Rubin Observatory site, calculated using
:class:`astroplan.Observer`.  The ``sun_alt`` parameter sets the solar altitude
threshold; science observations are generally expected from −12° twilight onward.

.. code-block:: python

    sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(20260605, sun_alt=-12)

Results are cached, so repeated calls for the same night are inexpensive.

To get sunset/sunrise for a range of nights as a DataFrame:

.. code-block:: python

    df = rn_dayobs.day_obs_sunset_sunrise_df(day_obs_min=20260101, day_obs_max=20260630)
    # columns: day_obs, sunset12, sunrise12, night_hours

Working with a range of nights
--------------------------------

:func:`~dayobs_utils.day_obs_list` returns every ``day_obs`` integer between two
:class:`~astropy.time.Time` values, exclusive of the last value:

.. code-block:: python

    from astropy.time import Time

    nights = rn_dayobs.day_obs_list(
        Time("2026-06-01T12:00:00"),
        Time("2026-06-10T12:00:00"),
    )
    # [20260601, 20260602, ..., 20260609]

Estimating visit counts
------------------------

:func:`~dayobs_utils.estimated_baseline_visit_range` estimates the expected number of
science visits for a night based on night length and the open shutter fraction from
baseline v5.1 survey simulations.  It returns a range rather than a single value to
reflect variability across simulations.

.. code-block:: python

    estimate = rn_dayobs.estimated_baseline_visit_range(20260605)
    # {"n_vis_ave": 840, "n_vis_high": 885}  (illustrative values)

    # Scale by expected open-shutter performance relative to baseline
    estimate = rn_dayobs.estimated_baseline_visit_range(20260605, relative_performance=0.9)
