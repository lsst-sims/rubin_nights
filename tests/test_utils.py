import unittest

import rubin_nights.plot_utils as rn_plots


class TestUtils(unittest.TestCase):

    def test_plot_styles(self) -> None:
        # Just test we get a dictionary with at least a band-color key
        band_colors = rn_plots.PlotStyles().band_colors
        self.assertTrue(
            list(band_colors.values()), ["#61A2B3", "#31DE1F", "#B52626", "#1600EA", "#BA52FF", "#370201"]
        )


if __name__ == "__main__":
    unittest.main()
