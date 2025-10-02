import unittest

from astropy.time import Time

import rubin_nights.dayobs_utils as rn_dayobs
import rubin_nights.plot_utils as rn_plots


class TestUtils(unittest.TestCase):
    def test_day_obs_basic(self):
        day_obs_int = 20250503
        day_obs_str = "2025-05-03"
        self.assertEqual(rn_dayobs.day_obs_int_to_str(day_obs_int), day_obs_str)
        self.assertEqual(rn_dayobs.day_obs_str_to_int(day_obs_str), day_obs_int)

    def test_day_obs_time(self):
        # Did day_obs_now return YYYY-MM-DD day_obs
        today = rn_dayobs.today_day_obs()
        self.assertTrue(isinstance(today, str))
        self.assertTrue("-" in today)

        yesterday = rn_dayobs.yesterday_day_obs()
        self.assertTrue(isinstance(yesterday, str))

        # Do we turn a given time into a day_obs as expected
        day_obs = "2025-05-03"
        expected_sunset = 60798.95802903221
        expected_sunrise = 60799.431616983835
        time = Time(expected_sunset, format="mjd", scale="tai")
        self.assertEqual(rn_dayobs.time_to_day_obs(time), day_obs)
        time = Time(expected_sunrise, format="mjd", scale="tai")
        self.assertEqual(rn_dayobs.time_to_day_obs(time), day_obs)

        # And day_obs into a time (at the start of the dayobs)
        day_obs_time = Time(f"{day_obs}T12:00:00", format="isot", scale="tai")
        self.assertEqual(rn_dayobs.day_obs_to_time(day_obs), day_obs_time)

        # For a given day_obs do we find sunset/sunrise correctly
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(day_obs, sun_alt=-12)
        self.assertAlmostEqual(expected_sunset, sunset.mjd)
        self.assertAlmostEqual(expected_sunrise, sunrise.mjd)
        # And if you provide day_obs as an int, does that work too
        intday_obs = int(day_obs.replace("-", ""))
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(intday_obs)
        self.assertAlmostEqual(expected_sunset, sunset.mjd)
        self.assertAlmostEqual(expected_sunrise, sunrise.mjd)
        # Can we change the sun altitude definition for sunrise
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(day_obs, sun_alt=0)
        self.assertTrue(sunset.mjd < expected_sunset)
        self.assertTrue(sunrise.mjd > expected_sunrise)

    def test_plot_styles(self):
        # Just test we get a dictionary with at least a band-color key
        band_colors = rn_plots.PlotStyles().band_colors
        self.assertTrue(
            list(band_colors.values()), ["#61A2B3", "#31DE1F", "#B52626", "#1600EA", "#BA52FF", "#370201"]
        )


if __name__ == "__main__":
    unittest.main()
