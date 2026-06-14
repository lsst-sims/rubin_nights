import datetime
import unittest

from astropy.time import Time, TimeDelta

import rubin_nights.dayobs_utils as rn_dayobs

# A fixed night used across tests to keep expected values stable.
DAY_OBS_STR = "2025-05-03"
DAY_OBS_INT = 20250503
# Noon TAI on day_obs — the canonical start-of-night Time.
DAY_OBS_TIME = Time("2025-05-03T12:00:00", format="isot", scale="tai")
# Known -12 deg sunset/sunrise MJD values for this night.
EXPECTED_SUNSET_MJD = 60798.95802903221
EXPECTED_SUNRISE_MJD = 60799.431616983835


class TestDayObsConversions(unittest.TestCase):

    def test_day_obs_str_int(self) -> None:
        self.assertEqual(rn_dayobs.day_obs_int_to_str(DAY_OBS_INT), DAY_OBS_STR)
        self.assertEqual(rn_dayobs.day_obs_str_to_int(DAY_OBS_STR), DAY_OBS_INT)

    def test_tomorrow_day_obs_is_string(self) -> None:
        tomorrow = rn_dayobs.tomorrow_day_obs()
        self.assertIsInstance(tomorrow, str)
        self.assertRegex(tomorrow, r"^\d{4}-\d{2}-\d{2}$")

    def test_tomorrow_is_one_day_after_today(self) -> None:
        today = rn_dayobs.today_day_obs()
        tomorrow = rn_dayobs.tomorrow_day_obs()
        today_int = rn_dayobs.day_obs_str_to_int(today)
        tomorrow_int = rn_dayobs.day_obs_str_to_int(tomorrow)
        # Check using 'date' to avoid month boundaries
        today_date = rn_dayobs.day_obs_to_date(today_int)
        tomorrow_date = rn_dayobs.day_obs_to_date(tomorrow_int)
        self.assertEqual((tomorrow_date - today_date).days, 1)

    def test_time_to_day_obs_int_returns_int(self) -> None:
        result = rn_dayobs.time_to_day_obs_int(DAY_OBS_TIME)
        self.assertIsInstance(result, int)
        self.assertEqual(result, DAY_OBS_INT)

    def test_time_to_day_obs_int_matches_str_conversion(self) -> None:
        t = Time("2025-11-27T03:00:00", format="isot", scale="utc")
        as_int = rn_dayobs.time_to_day_obs_int(t)
        as_str = rn_dayobs.time_to_day_obs(t)
        self.assertEqual(as_int, rn_dayobs.day_obs_str_to_int(as_str))

    def test_day_obs_to_date_from_int(self) -> None:
        result = rn_dayobs.day_obs_to_date(DAY_OBS_INT)
        self.assertIsInstance(result, datetime.date)
        self.assertEqual(result, datetime.date(2025, 5, 3))

    def test_day_obs_to_date_from_str(self) -> None:
        result = rn_dayobs.day_obs_to_date(DAY_OBS_STR)
        self.assertEqual(result, datetime.date(2025, 5, 3))

    def test_day_obs_to_date_from_int_str(self) -> None:
        # String form of the integer (no dashes) should also be accepted.
        result = rn_dayobs.day_obs_to_date(str(DAY_OBS_INT))
        self.assertEqual(result, datetime.date(2025, 5, 3))

    def test_roundtrip_int_str(self) -> None:
        self.assertEqual(
            rn_dayobs.day_obs_str_to_int(rn_dayobs.day_obs_int_to_str(DAY_OBS_INT)),
            DAY_OBS_INT,
        )

    def test_day_obs_to_time_accepts_string(self) -> None:
        t = rn_dayobs.day_obs_to_time(DAY_OBS_STR)
        self.assertEqual(t, DAY_OBS_TIME)

    def test_day_obs_to_time_accepts_int(self) -> None:
        t = rn_dayobs.day_obs_to_time(DAY_OBS_INT)
        self.assertEqual(t, DAY_OBS_TIME)

    def test_day_obs_time(self) -> None:
        # Did day_obs_now return YYYY-MM-DD day_obs
        today = rn_dayobs.today_day_obs()
        self.assertTrue(isinstance(today, str))
        self.assertTrue("-" in today)

        yesterday = rn_dayobs.yesterday_day_obs()
        self.assertTrue(isinstance(yesterday, str))

        # Do we turn a given time into a day_obs as expected
        time = Time(EXPECTED_SUNSET_MJD, format="mjd", scale="tai")
        self.assertEqual(rn_dayobs.time_to_day_obs(time), DAY_OBS_STR)
        time = Time(EXPECTED_SUNRISE_MJD, format="mjd", scale="tai")
        self.assertEqual(rn_dayobs.time_to_day_obs(time), DAY_OBS_STR)

        # And day_obs into a time (at the start of the dayobs)
        day_obs_time = Time(f"{DAY_OBS_STR}T12:00:00", format="isot", scale="tai")
        self.assertEqual(rn_dayobs.day_obs_to_time(DAY_OBS_STR), day_obs_time)

        # For a given day_obs do we find sunset/sunrise correctly
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(DAY_OBS_INT, sun_alt=-12)
        self.assertAlmostEqual(EXPECTED_SUNSET_MJD, sunset.mjd)
        self.assertAlmostEqual(EXPECTED_SUNRISE_MJD, sunrise.mjd)
        # And if you provide day_obs as an int, does that work too
        intday_obs = int(DAY_OBS_STR.replace("-", ""))
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(intday_obs)
        self.assertAlmostEqual(EXPECTED_SUNSET_MJD, sunset.mjd)
        self.assertAlmostEqual(EXPECTED_SUNRISE_MJD, sunrise.mjd)
        # Can we change the sun altitude definition for sunrise
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(DAY_OBS_INT, sun_alt=0)
        self.assertTrue(sunset.mjd < EXPECTED_SUNSET_MJD)
        self.assertTrue(sunrise.mjd > EXPECTED_SUNRISE_MJD)


class TestMjdToDayObs(unittest.TestCase):

    def test_mjd_during_night_gives_correct_day_obs(self) -> None:
        # MJD of midnight UTC during the 2025-05-03 night.
        midnight_mjd = Time("2025-05-04T00:00:00", format="isot", scale="tai").mjd
        result = rn_dayobs.mjd_to_dayobs(midnight_mjd)
        self.assertEqual(result, DAY_OBS_INT)

    def test_mjd_at_noon_start_of_day_obs(self) -> None:
        # Noon TAI on the day_obs date is the start of that day_obs.
        result = rn_dayobs.mjd_to_dayobs(DAY_OBS_TIME.mjd)
        self.assertEqual(result, DAY_OBS_INT)

    def test_mjd_just_before_noon_is_previous_day_obs(self) -> None:
        # 11:59 TAI on 2025-05-03 belongs to the previous night (2025-05-02).
        before_noon = Time("2025-05-03T11:59:00", format="isot", scale="tai").mjd
        result = rn_dayobs.mjd_to_dayobs(before_noon)
        self.assertEqual(result, 20250502)

    def test_mjd_returns_int(self) -> None:
        self.assertIsInstance(rn_dayobs.mjd_to_dayobs(DAY_OBS_TIME.mjd), int)


class TestDayObsList(unittest.TestCase):

    def test_single_night(self) -> None:
        t_start = rn_dayobs.day_obs_to_time(DAY_OBS_INT)
        t_end = t_start + TimeDelta(1, format="jd")
        result = rn_dayobs.day_obs_list(t_start, t_end)
        self.assertEqual(result, [DAY_OBS_INT])

    def test_three_consecutive_nights(self) -> None:
        t_start = rn_dayobs.day_obs_to_time(20250503)
        t_end = rn_dayobs.day_obs_to_time(20250505) + TimeDelta(1, format="jd")
        result = rn_dayobs.day_obs_list(t_start, t_end)
        self.assertEqual(result, [20250503, 20250504, 20250505])

    def test_returns_list_of_ints(self) -> None:
        t_start = rn_dayobs.day_obs_to_time(DAY_OBS_INT)
        t_end = t_start + TimeDelta(2, format="jd")
        result = rn_dayobs.day_obs_list(t_start, t_end)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(d, int) for d in result))

    def test_is_monotonically_increasing(self) -> None:
        t_start = rn_dayobs.day_obs_to_time(20250101)
        t_end = rn_dayobs.day_obs_to_time(20250110) + TimeDelta(1, format="jd")
        result = rn_dayobs.day_obs_list(t_start, t_end)
        self.assertTrue(all(a < b for a, b in zip(result, result[1:])))


class TestDayObsSunsetSunriseDF(unittest.TestCase):

    def test_returns_expected_columns(self) -> None:
        df = rn_dayobs.day_obs_sunset_sunrise_df(20250503, 20250505)
        for col in ("day_obs", "sunset12", "sunrise12", "night_hours"):
            self.assertIn(col, df.columns)

    def test_row_count_matches_nights(self) -> None:
        # day_obs_max HERE is inclusive, so 20250503–20250505
        # yields 20250503 and 20250504.
        df = rn_dayobs.day_obs_sunset_sunrise_df(20250503, 20250505)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["day_obs"]), [20250503, 20250504, 20250505])

    def test_night_hours_are_positive(self) -> None:
        df = rn_dayobs.day_obs_sunset_sunrise_df(20250503, 20250505)
        self.assertTrue((df["night_hours"] > 0).all())

    def test_sunset_before_sunrise(self) -> None:
        df = rn_dayobs.day_obs_sunset_sunrise_df(20250503, 20250503)
        self.assertTrue((df["sunset12"] < df["sunrise12"]).all())

    def test_known_night_hours(self) -> None:
        # Compute the expected night length from the known MJD values.
        expected_hours = (EXPECTED_SUNRISE_MJD - EXPECTED_SUNSET_MJD) * 24
        df = rn_dayobs.day_obs_sunset_sunrise_df(20250503, 20250503)
        self.assertAlmostEqual(df.iloc[0]["night_hours"], expected_hours, places=3)


class TestEstimatedBaselineVisitRange(unittest.TestCase):

    def test_returns_expected_keys(self) -> None:
        result = rn_dayobs.estimated_baseline_visit_range(DAY_OBS_INT)
        self.assertIn("n_vis_ave", result)
        self.assertIn("n_vis_high", result)

    def test_values_are_positive_ints(self) -> None:
        result = rn_dayobs.estimated_baseline_visit_range(DAY_OBS_INT)
        self.assertIsInstance(result["n_vis_ave"], int)
        self.assertIsInstance(result["n_vis_high"], int)
        self.assertGreater(result["n_vis_ave"], 0)
        self.assertGreater(result["n_vis_high"], 0)

    def test_high_is_at_least_average(self) -> None:
        result = rn_dayobs.estimated_baseline_visit_range(DAY_OBS_INT)
        self.assertGreaterEqual(result["n_vis_high"], result["n_vis_ave"])

    def test_relative_performance_scales_result(self) -> None:
        full = rn_dayobs.estimated_baseline_visit_range(DAY_OBS_INT, relative_performance=1.0)
        half = rn_dayobs.estimated_baseline_visit_range(DAY_OBS_INT, relative_performance=0.5)
        self.assertLess(half["n_vis_ave"], full["n_vis_ave"])
        self.assertLess(half["n_vis_high"], full["n_vis_high"])

    def test_winter_night_more_visits_than_summer(self) -> None:
        # June is winter (longest nights),
        # December is summer (shortest nights).
        winter = rn_dayobs.estimated_baseline_visit_range(20250621)
        summer = rn_dayobs.estimated_baseline_visit_range(20251221)
        self.assertGreater(winter["n_vis_ave"], summer["n_vis_ave"])


if __name__ == "__main__":
    unittest.main()
