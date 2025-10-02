import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Polygon

try:
    import healpy as hp
except ModuleNotFoundError:
    warnings.warn("Healpy used for some quick plots.")

__all__ = ["PlotStyles", "hp_laea", "hp_moll", "detector_plot"]

logger = logging.getLogger(__name__)


@dataclass
class PlotStyles:
    band_colors = {
        "u": "#1600EA",
        "g": "#31DE1F",
        "r": "#B52626",
        "i": "#370201",
        "z": "#BA52FF",
        "y": "#61A2B3",
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


def hp_laea(
    hp_array: np.ndarray,
    alpha: np.ndarray | None = None,
    label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    hp.azeqview(hp_array, alpha=alpha, rot=(0, -90, 0), lamb=True, reso=17.5, min=vmin, max=vmax, title=label)
    hp.graticule()


def hp_moll(
    hp_array: np.ndarray,
    alpha: np.ndarray | None = None,
    label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    hp.mollview(hp_array, alpha=alpha, min=vmin, max=vmax, title=label)
    hp.graticule()


def detector_plot(
    key: str,
    detector_values: pd.Series | pd.DataFrame,
    camera_df: pd.DataFrame,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Colormap | str = "viridis",
    text_color: str = "black",
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure | None, Axes]:
    """Plot the values per detector arranged across the focal plane.

    Parameters
    ----------
    key
        The value out of the `detValues` dataframe to plot.
    detector_values
        The dataframe with values per detector.
    camera_df
        A dataframe with camera detector locations, such
        as from `rubin_nights/data/lsstCamera.h5`.
    title
        Optional title for the plot.
    vmin, vmax
        The minimum and maximum values for the colorbar.
        If None, will use the nanmin/nanmax of the data.
    cmap
        Matplotlib colormap.
    test_color
        Color for the text over each detector.
    ax
        Matplotlib axes to use for the plot.


    Returns
    -------
    fig, ax : `matplotlib.Figure`, `matplotlib.axes.Axes`
        Matplotlib figure and axes for the plot.
    """
    fig: Figure | SubFigure
    if ax is None:
        fig = Figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
    else:
        tmpfig = ax.get_figure()
        if tmpfig is None:
            fig = Figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
        else:
            fig = tmpfig

    if vmin is None:
        vmin = np.nanmin(detector_values[key].values)
    if vmax is None:
        vmax = np.nanmax(detector_values[key].values)
    norm = Normalize(vmin=vmin, vmax=vmax)

    tmp = pd.merge(camera_df, detector_values, how="right", left_on="detId", right_on="detector")

    patches = []
    for det, row in tmp.iterrows():
        patches.append(Polygon(row.corners))
        ax.text(
            row.cenX,
            row.cenY,
            f"{row[key]:.2f}",
            ha="center",
            va="center",
            size="large",
            color=text_color,
            rotation=row.textRot,
        )

    patchCollection = PatchCollection(
        patches, edgecolor="black", cmap=cmap, linewidth=0.5, linestyle=(0, (0.5, 3))
    )
    patchCollection.set_array(tmp[key])
    ax.add_collection(patchCollection)

    median = np.nanmedian(tmp[key])
    mean = np.nanmean(tmp[key])
    std = np.nanstd(tmp[key])

    statsText = f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}"
    ax.text(
        0.95,
        0.95,
        statsText,
        transform=ax.transAxes,
        fontsize="large",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, aspect=30, shrink=0.8)
    cbar.set_label(f"{key}", fontsize="x-large")

    ax.set_xlabel("Field Angle Y [deg]", fontsize="large")
    ax.set_ylabel("Field Angle X [deg]", fontsize="large")
    ax.axis("equal")
    ax.grid(True, alpha=0.3, linestyle=":")

    if title:
        fig.suptitle(title, fontsize="x-large")

    if isinstance(fig, Figure):
        fig.tight_layout()
    return fig, ax
