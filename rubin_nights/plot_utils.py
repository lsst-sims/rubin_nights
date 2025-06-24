from dataclasses import dataclass


@dataclass
class PlotStyles:
    band_colors = {
        "u": "#61A2B3",
        "g": "#31DE1F",
        "r": "#B52626",
        "i": "#1600EA",
        "z": "#BA52FF",
        "y": "#370201",
    }
    band_symbols = {"u": "o", "g": "^", "r": "v", "i": "s", "z": "*", "y": "p"}
    band_linestyles = {
        "u": "--",
        "g": ":",
        "r": "-",
        "i": "-.",
        "z": (0, (3, 5, 1, 5, 1, 5)),
        "y": (0, (3, 1, 1, 1)),
    }
