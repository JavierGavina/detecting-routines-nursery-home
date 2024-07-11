import json
from typing import Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
import yaml

from src.DRFL import DRGS


# import argparse
#
# argparser = argparse.ArgumentParser()
# argparser.add_argument("--data_dir", type=str, default="data/activities-simulation.csv", help="Path to the data file")
# argparser.add_argument("--dictionary_dir", type=str, default="data/dictionary_rooms.json",
#                        help="Path to the dictionary file")
# argparser.add_argument("--param_m", type=int, default=4, help="length of the subsequences")
# argparser.add_argument("--param_R", type=int, default=10, help="least maximum distance between subsequences")
# argparser.add_argument("--param_C", type=int, default=4, help="minimum number of matches of a routine")
# argparser.add_argument("--param_G", type=int, default=60, help="minimum magnitude of a subsequence")
# argparser.add_argument("--epsilon", type=float, default=0.5, help="minimum overlap percentage")
# argparser.add_argument("--L", type=int, default=0, help="minimum number of subsequences in a routine")
# argparser.add_argument("--fusion_distance", type=float, default=0.001,
#                        help="minimum distance between clusters centroids to be fused")



def plot_quarters_groundtruth(*, time_series: pd.Series,
                              room: str,
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
    if isinstance(barcolors, list):
        barcolors = np.array(barcolors)/255

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
        ax.set_xticks(np.arange(0, 24 * 4, 2), labels=[x for idx, x in enumerate(x_hour_minutes) if idx % 2 == 0],
                      rotation=90, fontsize=ticks_size)
        ax.set_yticks(np.arange(0, 19, 4), fontsize=ticks_size)
        if show_grid:
            ax.grid(True)

        ax.set_ylim(0, 19)
        if xlim:
            ax.set_xlim(st_idx - 1, en_idx + 1)
        else:
            ax.set_xlim(-1, 24 * 4 + 1)

        # Annotate height of the bar
        for idx, value in enumerate(time_series[i * 24 * 4:(i + 1) * 24 * 4]):
            if xlim:
                if st_idx <= idx <= en_idx:
                    ax.text(idx, value + 0.5, str(value), ha='center', va='bottom', fontsize=ticks_size)

            else:
                ax.text(idx, value + 0.5, str(value), ha='center', va='bottom', fontsize=ticks_size)

    plt.tight_layout()

    if save_dir is not None:
        format = save_dir.split(".")[-1]
        plt.savefig(save_dir, format=format)

    if show_plot:
        plt.show()

    plt.close()


def get_path_figs(dificulty: str):
    if not isinstance(dificulty, str):
        raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

    return f"figs/Groundtruth Synthetic Activity Dataset/402E/{dificulty}/quarters"


def get_path_result(dificulty: str):
    if not isinstance(dificulty, str):
        raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

    return f"results/Synthetic Activity Dataset/402E/{dificulty}/plot_quarters_routines/Therapy Room"


class DataLoaderUser:
    def __init__(self, data_dir:str, user: str, dificulty: str):
        self.__check_params(data_dir, user, dificulty)
        self.data_dir = data_dir
        self.user = user
        self.dificulty = dificulty
        self.path = self.__get_path()

    def __get_path(self):
        path = f"{self.data_dir}/{self.user}/{self.dificulty}/out_feat_extraction_quarters.csv"
        return path

    @staticmethod
    def __check_params(data_dir: str, user: str, dificulty: str):
        if not isinstance(dificulty, str):
            raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

        if not isinstance(user, str):
            raise TypeError(f"User must be a string. Got {type(user).__name__}")

        if not isinstance(data_dir, str):
            raise TypeError(f"Data dir must be a string. Got {type(data_dir).__name__}")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} not found")

        if not os.path.exists(f"{data_dir}/{user}/{dificulty}/out_feat_extraction_quarters.csv"):
            raise FileNotFoundError(f"{data_dir}/{user}/{dificulty}/out_feat_extraction_quarters.csv not found")

    def __get_time_series(self, room: str, select_month: Optional[int] = 3):

        if not isinstance(room, str):
            raise TypeError(f"Room must be a string. Got {type(room).__name__}")

        if not isinstance(select_month, int) and select_month is not None:
            raise TypeError(f"Select month must be an integer or None. Got {type(select_month).__name__}")

        feat_extraction = pd.read_csv(self.path)
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])
        for row in range(feat_extraction.shape[0]):
            feat_extraction.loc[row, "Date"] = feat_extraction.loc[row, "Date"] + pd.Timedelta(
                minutes=feat_extraction.loc[row, "Quarter"] * 15)

        feat_extraction.set_index("Date", inplace=True)

        if select_month is not None:
            feat_extraction = feat_extraction[feat_extraction["Month"] == int(select_month)]

        room_time_series = feat_extraction[f"N_{room}"]

        return room_time_series

    def load_time_series(self, room: str, select_month: Optional[int] = 3):
        return self.__get_time_series(room=room, select_month=select_month)


def setup():
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":
    config = setup()
    # 'data_dir', 'users', 'dificulties', 'rooms', 'colors', 'params', 'intervals_of_interest'
    print(config["intervals_of_interest"]["9FE9"])
    # for user in config["users"]:
    #     for dificulty in config["dificulties"]:
    #         data_loader = DataLoaderUser(data_dir=config["data_dir"], user=user, dificulty=dificulty)
    #         for room in config["rooms"]:
    #             time_series = data_loader.load_time_series(room=room)
    #             plot_quarters_groundtruth(time_series=time_series, room=room, barcolors=config["colors"][room], save_dir=None)
                # params = config["params"]["Room"] if room == "Room" else config["params"]["other"]
                # drfl = DRGS(**params)
                # drfl.fit(time_series=time_series, verbose=False)
                # drfl.




