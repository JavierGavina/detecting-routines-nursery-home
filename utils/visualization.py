"""
This module contains functions to visualize the results of the DRGS model and the ground truth of the activities in the room.

Public Functions:
----------------

- ``plot_quarters_groundtruth``: Plot the time series of a room in quarters with the ground truth of the activities
- ``save_results_plot``: Save the results of the DRGS model in a directory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import warnings
import os
import pandas as pd

sys.path.append("..")
from src.DRFL import DRGS
from typing import Union, Optional


def plot_quarters_groundtruth(*, time_series: pd.Series,
                              top_days: int = 30,
                              figsize: tuple[int, int] = (30, 30),
                              title_size: int = 20,
                              ticks_size: int = 10,
                              labels_size: int = 15,
                              barcolors: Union[str, np.ndarray, list[int, int, int]] = "blue",
                              linewidth: Union[int, float] = 1.5,
                              show_plot: bool = True,
                              show_grid: bool = True,
                              xlim: Optional[tuple[str, str]] = None,
                              save_dir: Optional[str] = None):
    """
    Plot the time series of a room in quarters with the ground truth of the activities

    Parameters:
        time_series: ``pd.Series``: Time series of the room
        top_days: ``int``: Number of days to plot
        figsize: ``tuple[int, int]``: Size of the figure. Default (30, 30)
        title_size: ``int``: Size of the title. Default 20
        ticks_size: ``int``: Size of the ticks. Default 10
        labels_size: ``int``: Size of the labels. Default 15
        barcolors: ``Union[str, np.ndarray, list[int, int, int]]``: Color of the bars. Default "blue"
        linewidth: ``Union[int, float]``: Width of the lines. Default 1.5
        show_plot: ``bool``: Show the plot. Default True
        show_grid: ``bool``: Show the grid. Default True
        xlim: ``Optional[tuple[str, str]]``: Limit of the x axis. Default None
        save_dir: ``Optional[str]``: Path to save the plot. Default None
    """

    if isinstance(barcolors, list):
        barcolors = np.array(barcolors) / 255

    date = time_series.index
    top_days = min(top_days, len(date) // (24 * 4))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(top_days, 1, figure=fig)

    if xlim:
        start_hour, end_hour = xlim
        st_h, st_m = start_hour.split(":")
        en_h, en_m = end_hour.split(":")
        st_h, st_m, en_h, en_m = int(st_h), int(st_m), int(en_h), int(en_m)
        st_idx = st_h * 4 + st_m // 15
        en_idx = en_h * 4 + en_m // 15

    for i in range(top_days):
        x_hour_minutes = [f"{hour:02}:{minute:02}" for hour in range(24) for minute in range(0, 60, 15)]
        ax = fig.add_subplot(gs[i, 0])
        ax.bar(np.arange(0, 24 * 4, 1), time_series[i * 24 * 4:(i + 1) * 24 * 4],
               color=barcolors, edgecolor="black", linewidth=linewidth)
        weekday = date[i * 24 * 4].weekday()
        weekday_str = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]

        ax.set_title(
            f"Date {date[i * 24 * 4].year} / {date[i * 24 * 4].month} / {date[i * 24 * 4].day}; Weekday: {weekday_str}",
            fontsize=title_size
        )

        ax.set_xlabel("Time", fontsize=labels_size)
        ax.set_ylabel("N minutes", fontsize=labels_size)

        ax.set_xticks(ticks=np.arange(0, 24 * 4, 2), labels=[x for idx, x in enumerate(x_hour_minutes) if idx % 2 == 0],
                      rotation=90, fontsize=ticks_size)
        ax.set_yticks(ticks=np.arange(0, 19, 4), labels=np.arange(0, 19, 4), fontsize=ticks_size)

        if show_grid:
            ax.grid(True)

        ax.set_ylim(0, 19)
        if xlim:
            ax.set_xlim(st_idx - 1, en_idx + 1)
        else:
            ax.set_xlim(-1, 24 * 4 + 1)

        # Annotate height of the bar
        for idx, value in enumerate(time_series[i * 24 * 4:(i + 1) * 24 * 4]):
            if (xlim and st_idx <= idx <= en_idx) or not xlim:
                ax.text(idx, value + 0.5, str(value), ha='center', va='bottom', fontsize=ticks_size)

    plt.tight_layout()

    if save_dir is not None:
        format = save_dir.split(".")[-1]
        plt.savefig(save_dir, format=format)

    if show_plot:
        plt.show()

    plt.close()


def save_results_plot(*,
                      drgs_fitted: DRGS,
                      room_routines: str,
                      user: str,
                      dificulty: str,
                      room: str,
                      figsize_cluster: tuple[int, int] = (15, 23),
                      figsize_tree: tuple[int, int] = (18, 7),
                      show_plot: bool = False,
                      format: str = "pdf",
                      top_days: int = 7):

    """
    Save the results of the DRGS model in a directory

    Parameters:
        drgs_fitted: ``DRGS``: Fitted DRGS model
        room_routines: ``str``: Path to save the results
        user: ``str``: User of the data
        dificulty: ``str``: Dificulty of the data
        room: ``str``: Room of the data
        figsize_cluster: ``tuple[int, int]``: Size of the cluster plot. Default (15, 23)
        figsize_tree: ``tuple[int, int]``: Size of the tree plot. Default (18, 7)
        show_plot: ``bool``: Show the plot. Default False
        format: ``str``: Format to save the plot. Default "pdf"
        top_days: ``int``: Number of days to plot. Default 7

    Raises:
        ValueError: If the format is not one of pdf, png or svg or if the DRGS model is not fitted
    """

    if format not in ["pdf", "png", "svg"]:
        raise ValueError(f"Format must be one of pdf, png or svg. Got {format}")

    if not drgs_fitted.is_fitted():
        raise ValueError("DRGS must be fitted to save the results")

    detected_routines = drgs_fitted.get_results()

    if detected_routines.is_empty():
        warnings.warn(f"No routines detected for user {user}, room {room} and dificulty {dificulty}")
        return

    path_out = f"{room_routines}/{room}"
    os.makedirs(path_out, exist_ok=True)
    xlim = ("09:30", "20:00") if room != "Room" else None
    drgs_fitted.results_per_quarter_hour(top_days=top_days, figsize=figsize_cluster, save_dir=path_out,
                                         bars_linewidth=2, show_background_annotations=True,
                                         show_plot=show_plot, format=format, xlim=xlim)

    tree = drgs_fitted.convert_to_cluster_tree()
    tree.plot_tree(title=f"Result Tree dificulty {dificulty}",
                   show_plot=show_plot,
                   save_dir=f"{path_out}/final_tree_quarters.{format}",
                   figsize=figsize_tree)
