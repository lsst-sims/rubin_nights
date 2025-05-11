import unittest

from astropy.time import Time

import rubin_nights.utils as utils


class TestUtils(unittest.TestCase):
    def test_day_obs_basic(self):
        day_obs_int = 20250503
        day_obs_str = "2025-05-03"
        self.assertEqual(utils.day_obs_int_to_str(day_obs_int), day_obs_str)
        self.assertEqual(utils.day_obs_str_to_int(day_obs_str), day_obs_int)

    def test_day_obs_time(self):
        # Did day_obs_now return YYYY-MM-DD day_obs
        today = utils.today_day_obs()
        self.assertTrue(isinstance(today, str))
        self.assertTrue("-" in today)
        yesterday = utils.yesterday_day_obs()
        self.assertTrue(isinstance(yesterday, str))
        # Do we turn a given time into a day_obs as expected
        day_obs = "2025-05-03"
        expected_sunset = 60798.9377821316
        expected_sunrise = 60799.45103942463
        time = Time(expected_sunset, format="mjd", scale="utc")
        self.assertEqual(utils.time_to_day_obs(time), day_obs)
        time = Time(expected_sunrise, format="mjd", scale="utc")
        self.assertEqual(utils.time_to_day_obs(time), day_obs)
        # For a given day_obs do we find sunset/sunrise correctly
        sunset, sunrise = utils.day_obs_sunset_sunrise(day_obs)
        self.assertAlmostEqual(expected_sunset, sunset.mjd)
        self.assertAlmostEqual(expected_sunrise, sunrise.mjd)
        # And if you provide day_obs as an int, does that work too
        intday_obs = int(day_obs.replace("-", ""))
        sunset, sunrise = utils.day_obs_sunset_sunrise(intday_obs)
        self.assertAlmostEqual(expected_sunset, sunset.mjd)
        self.assertAlmostEqual(expected_sunrise, sunrise.mjd)

    def test_plot_styles(self):
        # Just test we get a dictionary with at least a band-color key
        plot_styles = utils.rtn045_plot_styles()
        keys = list(plot_styles.keys())
        self.assertTrue(len([c for c in keys if "band_colors" in c]) > 0)


if __name__ == "__main__":
    unittest.main()
