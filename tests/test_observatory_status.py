import unittest
from unittest.mock import Mock

import pandas as pd
from astropy.time import Time

from rubin_nights.observatory_status import get_dome_open_close

# Use a real night so sunset/sunrise calculations are grounded.
DAY_OBS = 20250503
T_START = Time("2025-05-03T12:00:00", format="isot", scale="utc")
T_END = Time("2025-05-04T12:00:00", format="isot", scale="utc")

# Dome open/close times well within the night (UTC).
OPEN_TIME = pd.Timestamp("2025-05-03 23:30:00", tz="UTC")
CLOSE_TIME = pd.Timestamp("2025-05-04 08:00:00", tz="UTC")

# A second open/close pair on the same night (>5 min gap from first open).
OPEN_TIME_2 = pd.Timestamp("2025-05-04 05:00:00", tz="UTC")
CLOSE_TIME_2 = pd.Timestamp("2025-05-04 07:00:00", tz="UTC")


def _make_shutter_df(timestamps: list) -> pd.DataFrame:
    """Return a minimal apertureShutter DataFrame with a UTC DatetimeIndex."""
    index = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.DataFrame(
        {
            "positionActual0": [50.0] * len(timestamps),
            "positionActual1": [50.0] * len(timestamps),
            "positionCommanded0": [100.0] * len(timestamps),
            "positionCommanded1": [100.0] * len(timestamps),
        },
        index=index,
    )


def _make_efd_client(open_df: pd.DataFrame, close_df: pd.DataFrame) -> Mock:
    """Return a mock EFD client whose .query()
    returns open_df then close_df."""
    client = Mock()
    client.query.side_effect = [open_df, close_df]
    return client


class TestGetDomeOpenClose(unittest.TestCase):

    def test_no_dome_events_returns_nat_row(self) -> None:
        """When the dome never opened, each day_obs gets a NaT row."""
        client = _make_efd_client(pd.DataFrame([]), pd.DataFrame([]))
        result = get_dome_open_close(T_START, T_END, client, with_sunset_sunrise=False)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["day_obs"], DAY_OBS)
        self.assertTrue(pd.isna(result.iloc[0]["open_time"]))
        self.assertTrue(pd.isna(result.iloc[0]["close_time"]))
        self.assertEqual(result.iloc[0]["dome_hours"], 0)

    def test_single_open_close_pair(self) -> None:
        """A matched open/close pair produces the correct dome_hours."""
        open_df = _make_shutter_df([OPEN_TIME])
        close_df = _make_shutter_df([CLOSE_TIME])
        client = _make_efd_client(open_df, close_df)

        result = get_dome_open_close(T_START, T_END, client, with_sunset_sunrise=False)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["day_obs"], DAY_OBS)
        self.assertAlmostEqual(
            row["dome_hours"],
            (CLOSE_TIME - OPEN_TIME) / pd.Timedelta(1, "h"),
            places=4,
        )

    def test_open_without_close_gives_nat_close(self) -> None:
        """An open event with no following close should
        return NaT for close_time and time up to now/sunrise for dome_hours."""
        open_df = _make_shutter_df([OPEN_TIME])
        client = _make_efd_client(open_df, pd.DataFrame([]))

        result = get_dome_open_close(T_START, T_END, client, with_sunset_sunrise=False)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertFalse(pd.isna(row["open_time"]))
        self.assertTrue(pd.isna(row["close_time"]))
        self.assertTrue(row["dome_hours"] > 0)

    def test_multiple_open_close_pairs_same_night(self) -> None:
        """Two open/close pairs separated by >5 minutes each produce a row."""
        # First pair: 23:30 open → 04:00 close
        # Second pair: 05:00 open → 07:00 close (>5 min gap from first open)
        close_time_1 = pd.Timestamp("2025-05-04 04:00:00", tz="UTC")
        open_df = _make_shutter_df([OPEN_TIME, OPEN_TIME_2])
        close_df = _make_shutter_df([close_time_1, CLOSE_TIME_2])
        client = _make_efd_client(open_df, close_df)

        result = get_dome_open_close(T_START, T_END, client, with_sunset_sunrise=False)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(result["day_obs"] == DAY_OBS))
        # First open matched to close_time_1.
        self.assertAlmostEqual(
            result.iloc[0]["dome_hours"],
            (close_time_1 - OPEN_TIME) / pd.Timedelta(1, "h"),
            places=4,
        )
        # Second open matched to CLOSE_TIME_2.
        self.assertAlmostEqual(
            result.iloc[1]["dome_hours"],
            (CLOSE_TIME_2 - OPEN_TIME_2) / pd.Timedelta(1, "h"),
            places=4,
        )

    def test_with_sunset_sunrise_adds_columns(self) -> None:
        """with_sunset_sunrise=True adds
        sunset12, sunrise12, night_hours, open_hours."""
        open_df = _make_shutter_df([OPEN_TIME])
        close_df = _make_shutter_df([CLOSE_TIME])
        client = _make_efd_client(open_df, close_df)

        result = get_dome_open_close(T_START, T_END, client, with_sunset_sunrise=True)

        for col in ("sunset12", "sunrise12", "night_hours", "open_hours"):
            self.assertIn(col, result.columns)

        row = result.iloc[0]
        self.assertGreater(row["night_hours"], 0)
        self.assertGreater(row["open_hours"], 0)
        # open_hours must be <= night_hours (dome can't be open
        # longer than the night).
        self.assertLessEqual(row["open_hours"], row["night_hours"])

    def test_without_sunset_sunrise_omits_columns(self) -> None:
        """with_sunset_sunrise=False leaves out the night-hours columns."""
        client = _make_efd_client(pd.DataFrame([]), pd.DataFrame([]))
        result = get_dome_open_close(T_START, T_END, client, with_sunset_sunrise=False)

        for col in ("sunset12", "sunrise12", "night_hours", "open_hours"):
            self.assertNotIn(col, result.columns)

    def test_multi_night_range(self) -> None:
        """A two-night range with no events returns one row per night."""
        t_end_2 = Time("2025-05-05T12:00:00", format="isot", scale="utc")
        client = _make_efd_client(pd.DataFrame([]), pd.DataFrame([]))

        result = get_dome_open_close(T_START, t_end_2, client, with_sunset_sunrise=False)

        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["day_obs"]), [20250503, 20250504])


if __name__ == "__main__":
    unittest.main()
