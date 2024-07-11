import json
from typing import Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os

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


def extract_data_grouped_by_hour(path_to_data: str, path_to_dictionary: str) -> pd.DataFrame:
    data = pd.read_csv(path_to_data, skiprows=1, header=None)
    correspondencies = json.load(open(path_to_dictionary))
    data.columns = ["Year", "Month", "Day"] + [f"Sequence_{x}" for x in range(1439)]
    out_columns = ["Year", "Month", "Day", "Hour"] + [f"N_{x}" for x in correspondencies.keys()]
    new_df = pd.DataFrame(columns=out_columns)
    for row in range(data.shape[0]):
        for hour in range(24):
            sequence = data.iloc[row, 3 + hour * 60:3 + (hour + 1) * 60].replace(np.nan, 0).values
            frequency_rooms = [np.sum(sequence == i) for i in correspondencies.values()]
            year, month, day = data.iloc[row, 0], data.iloc[row, 1], data.iloc[row, 2]
            new_df.loc[len(new_df)] = [year, month, day, hour] + frequency_rooms

    return new_df


def extract_data_grouped_by_quarter_hour(path_to_data: str, path_to_dictionary: str) -> pd.DataFrame:
    data = pd.read_csv(path_to_data, skiprows=1, header=None)
    correspondencies = json.load(open(path_to_dictionary))
    data.columns = ["Year", "Month", "Day"] + [f"Sequence_{x}" for x in range(1439)]
    out_columns = ["Year", "Month", "Day", "Hour", "Quarter"] + [f"N_{x}" for x in correspondencies.keys()]
    new_df = pd.DataFrame(columns=out_columns)
    for row in range(data.shape[0]):
        for hour in range(24):
            for quarter in range(4):
                sequence = data.iloc[row, 3 + hour * 60 + quarter * 15:3 + hour * 60 + (quarter + 1) * 15].replace(
                    np.nan, 0).values
                frequency_rooms = [np.sum(sequence == i) for i in correspondencies.values()]
                year, month, day = data.iloc[row, 0], data.iloc[row, 1], data.iloc[row, 2]
                new_df.loc[len(new_df)] = [year, month, day, hour, quarter] + frequency_rooms

    return new_df


def get_time_series(path_to_feat_extraction: str, room: str, select_month: Optional[str] = None) -> pd.Series:
    feat_extraction = pd.read_csv(path_to_feat_extraction)
    if "Quarter" not in feat_extraction.columns.tolist():
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])

    else:
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])
        for row in range(feat_extraction.shape[0]):
            feat_extraction.loc[row, "Date"] = feat_extraction.loc[row, "Date"] + pd.Timedelta(
                minutes=feat_extraction.loc[row, "Quarter"] * 15)

    feat_extraction.set_index("Date", inplace=True)
    if select_month is not None:
        feat_extraction = feat_extraction[feat_extraction["Month"] == int(select_month)]

    room_time_series = feat_extraction[f"N_{room}"]

    return room_time_series


def plot_quarters_groundtruth(*, time_series: pd.Series,
                              room: str,
                              top_days: int = 30,
                              figsize: tuple[int, int] = (30, 30),
                              title_size: int = 20,
                              ticks_size: int = 10,
                              labels_size: int = 15,
                              barcolors: Union[str, np.ndarray, tuple[int, int, int]] = "blue",
                              linewidth: Union[int, float] = 1.5,
                              show_plot: bool = True,
                              show_grid: bool = True,
                              xlim: Optional[tuple[str, str]] = None,
                              save_dir: Optional[str] = None):
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


def get_path(dificulty: str):
    if not isinstance(dificulty, str):
        raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

    if dificulty not in ["easy", "medium", "hard"]:
        raise ValueError(f"Dificulty must be one of 'easy', 'medium' or 'hard'. Got {dificulty}")

    return f"data/Synthetic Activity Dataset/402E/{dificulty}/out_feat_extraction_quarters.csv"


def get_path_figs(dificulty: str):
    if not isinstance(dificulty, str):
        raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

    if dificulty not in ["easy", "medium", "hard"]:
        raise ValueError(f"Dificulty must be one of 'easy', 'medium' or 'hard'. Got {dificulty}")

    return f"figs/Groundtruth Synthetic Activity Dataset/402E/{dificulty}/quarters"


def get_path_result(dificulty: str):
    if not isinstance(dificulty, str):
        raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

    if dificulty not in ["easy", "medium", "hard"]:
        raise ValueError(f"Dificulty must be one of 'easy', 'medium' or 'hard'. Got {dificulty}")

    return f"results/Synthetic Activity Dataset/402E/{dificulty}/plot_quarters_routines/Therapy Room"


if __name__ == "__main__":
    ROOT_DATA = "data/Synthetic Activity Dataset"
    RESULTS_PATH = "results/Synthetic Activity Dataset"
    FIGS_PATH = "figs/Groundtruth Synthetic Activity Dataset"

    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(FIGS_PATH, exist_ok=True)

    ROOM = "Therapy Room"

    easy = get_time_series(path_to_feat_extraction=get_path("easy"), room=ROOM,
                           select_month="3")
    medium = get_time_series(path_to_feat_extraction=get_path("medium"), room=ROOM,
                             select_month="3")
    hard = get_time_series(path_to_feat_extraction=get_path("hard"), room=ROOM,
                           select_month="3")

    plot_quarters_groundtruth(time_series=easy, room=ROOM,
                              barcolors=np.array([(0, 0, 255)]) / 255,
                              top_days=7, figsize=(7, 10), linewidth=2,
                              title_size=10, ticks_size=6, labels_size=7,
                              xlim=("10:00", "19:30"), show_grid=False,
                              save_dir=f"{get_path_figs('easy')}/{ROOM}.png", show_plot=True)

    plot_quarters_groundtruth(time_series=medium, room=ROOM,
                              barcolors=np.array([(0, 0, 255)]) / 255,
                              top_days=7, figsize=(7, 10), linewidth=2,
                              title_size=10, ticks_size=6, labels_size=7,
                              xlim=("10:00", "19:30"), show_grid=False,
                              save_dir=f"{get_path_figs('medium')}/{ROOM}.png", show_plot=True)

    plot_quarters_groundtruth(time_series=hard, room=ROOM,
                              barcolors=np.array([(0, 0, 255)]) / 255,
                              top_days=7, figsize=(7, 10), linewidth=2,
                              title_size=10, ticks_size=6, labels_size=7,
                              xlim=("10:00", "19:30"), show_grid=False,
                              save_dir=f"{get_path_figs('hard')}/{ROOM}.png", show_plot=True)

    epsilon, fusion_distance = 0.5, 0.001

    R, C, G, L = 3, 10, 8, 1
    R1, C1, G1, L1 = 8, 25, 8, 2
    R2, C2, G2, L2 = 8, 15, 8, 3

    easy_drfl = DRGS(length_range=(3, 100), R=R, C=C, G=G, L=L, epsilon=epsilon, fusion_distance=fusion_distance)
    medium_drfl = DRGS(length_range=(3, 100), R=R1, C=C1, G=G1, L=L1, epsilon=epsilon, fusion_distance=fusion_distance)
    hard_drfl = DRGS(length_range=(3, 100), R=R2, C=C2, G=G2, L=L2, epsilon=epsilon, fusion_distance=fusion_distance)

    easy_drfl.fit(easy, verbose=False)
    medium_drfl.fit(medium, verbose=False)
    hard_drfl.fit(hard, verbose=False)

    easy_tree = easy_drfl.convert_to_cluster_tree()
    medium_tree = medium_drfl.convert_to_cluster_tree()
    hard_tree = hard_drfl.convert_to_cluster_tree()

    os.makedirs(get_path_result("easy"), exist_ok=True)
    os.makedirs(get_path_result("medium"), exist_ok=True)
    os.makedirs(get_path_result("hard"), exist_ok=True)

    easy_drfl.results_per_quarter_hour(top_days=7, figsize=(7, 10), save_dir=get_path_result("easy"),
                                       title_fontsize=10, coloured_text_fontsize=6, text_fontsize=6,
                                       labels_fontsize=7, bars_linewidth=2, show_background_annotations=True,
                                       show_plot=False, format="svg", xlim=("10:00", "19:30"))

    medium_drfl.results_per_quarter_hour(top_days=7, figsize=(7, 10), save_dir=get_path_result("medium"),
                                         title_fontsize=10, coloured_text_fontsize=6, text_fontsize=6,
                                         labels_fontsize=7, bars_linewidth=2, show_background_annotations=True,
                                         show_plot=False, format="svg", xlim=("10:00", "19:30"))

    hard_drfl.results_per_quarter_hour(top_days=7, figsize=(7, 10), save_dir=get_path_result("hard"),
                                       title_fontsize=10, coloured_text_fontsize=6, text_fontsize=6,
                                       labels_fontsize=7, bars_linewidth=2, show_background_annotations=True,
                                       show_plot=False, format="svg", xlim=("10:00", "19:30"))

    easy_tree.plot_tree(save_dir=f"{get_path_result('easy')}/easy_tree.svg", title="Grafo crecimiento Groundtruth")
    medium_tree.plot_tree(save_dir=f"{get_path_result('medium')}/medium_tree.svg", title="Grafo crecimiento medio")
    hard_tree.plot_tree(save_dir=f"{get_path_result('hard')}/hard_tree.svg", title="Grafo crecimiento dificil")





    # for USER in ["02A8", "9FE9", "52EA", "402E", "682A", "F176"]:
    # for USER in ["402E"]:
    #     st = time.time()
    #     for DIFICULTY in ["easy", "medium", "hard"]:
    #         DICTIONARY_FILE = f"{ROOT_DATA}/{USER}/metadata/dictionary_rooms.json"
    #         DIFICULTY_DATA = f"{ROOT_DATA}/{USER}/{DIFICULTY}"
    #         DATA_FILE = f"{DIFICULTY_DATA}/activities-simulation.csv"
    #
    #         QUARTER_EXTRACTED = f"{DIFICULTY_DATA}/out_feat_extraction_quarters.csv"
    #
    #         # Results figs
    #         RESULTS_FIG = f"{RESULTS_PATH}/{USER}"
    #         RESULTS_FIG_DIFICULTY = f"{RESULTS_FIG}/{DIFICULTY}"
    #         QUARTER_RESULTS = f"{RESULTS_FIG_DIFICULTY}/plot_quarters_routines"
    #
    #         # Create paths for results figs
    #         os.makedirs(RESULTS_FIG, exist_ok=True)
    #         os.makedirs(RESULTS_FIG_DIFICULTY, exist_ok=True)
    #         os.makedirs(QUARTER_RESULTS, exist_ok=True)
    #
    #         # Groundtruth figs
    #         GROUNDTRUTH_ROOT = f"{FIGS_PATH}/{USER}"
    #         GROUNDTRUTH_DIFICULTY = f"{GROUNDTRUTH_ROOT}/{DIFICULTY}"
    #         QUARTERS_GROUNDTRUTH = f"{GROUNDTRUTH_DIFICULTY}/quarters"
    #
    #         # Create paths for groundtruth figs
    #         os.makedirs(GROUNDTRUTH_ROOT, exist_ok=True)
    #         os.makedirs(GROUNDTRUTH_DIFICULTY, exist_ok=True)
    #         os.makedirs(QUARTERS_GROUNDTRUTH, exist_ok=True)
    #
    #         correspondencies = json.load(open(DICTIONARY_FILE))
    #
    #         quarter_data = extract_data_grouped_by_quarter_hour(DATA_FILE, DICTIONARY_FILE)
    #
    #         quarter_data.to_csv(QUARTER_EXTRACTED, index=False)
    #
    #         all_rooms = list(correspondencies.keys())
    #         for _, room in tqdm(enumerate(["Therapy Room"])):
    #             id_room = 2
    #             st2 = time.time()
    #             path_out_quarter = f"{QUARTER_RESULTS}/{room}"
    #
    #             quarter_time_series = get_time_series(path_to_feat_extraction=QUARTER_EXTRACTED, room=room,
    #                                                   select_month="3")
    #
    #             if np.sum(quarter_time_series) == 0:
    #                 warnings.warn(f"Empty time series for room {room}")
    #                 continue
    #
    #             os.makedirs(path_out_quarter, exist_ok=True)
    #
    #             # Obtain rainbow colors
    #             customized_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
    #                                  (255, 255, 0), (128, 0, 128), (0, 255, 255),
    #                                  (255, 165, 0), (144, 238, 144)]
    #
    #             plot_quarters_groundtruth(time_series=quarter_time_series, room=room,
    #                                       barcolors=np.array(customized_colors[id_room]) / 255,
    #                                       top_days=7, figsize=(15, 23), linewidth=2,
    #                                       xlim = ("09:30", "20:00"), show_grid=False,
    #                                       save_dir=f"{QUARTERS_GROUNDTRUTH}/{room}.pdf", show_plot=False)
    #
    #             R2, C2, G2 = 3, 10, 8  # Quarters params
    #
    #             drgs_quarters = DRGS(length_range=(3, 100), R=R2, C=C2, G=G2, epsilon=0.5, L=1, fusion_distance=0.0001)
    #
    #             drgs_quarters.fit(quarter_time_series)
    #             drgs_quarters.results_per_quarter_hour(top_days=7, figsize=(15, 23), save_dir=path_out_quarter,
    #                                                    bars_linewidth=2, show_background_annotations=True,
    #                                                    show_plot=False, format="pdf", xlim=("09:30", "20:00"))
    #
    #             tree_quarters = drgs_quarters.convert_to_cluster_tree()
    #
    #             if drgs_quarters.get_results().is_empty():
    #                 warnings.warn(f"Empty results for room {room}")
    #
    #             figsize = (18, 7)
    #             if DIFICULTY == "easy":
    #                 title_tree = "Groundtruth Tree"
    #             else:
    #                 title_tree = f"Result Tree dificulty {DIFICULTY}"
    #
    #             tree_quarters.plot_tree(title=title_tree,
    #                                     show_plot=False,
    #                                     save_dir=f"{path_out_quarter}/final_tree_quarters.pdf",
    #                                     figsize=figsize)
    #
    #             print(f"Elapsed time for room {room}: {time.time() - st2}")
    #     print(f"Elapsed time for user {USER} and difficulty {DIFICULTY}: {time.time() - st}")

    # for DIFICULTY in ["easy", "medium", "hard"]:
    #
    #     ROOT_DATA = "data/data2"
    #     DICTIONARY_FILE = f"{ROOT_DATA}/metadata/dictionary_rooms.json"
    #     DIFICULTY_DATA = f"{ROOT_DATA}/{DIFICULTY}"
    #     DATA_FILE = f"{DIFICULTY_DATA}/activities-simulation-{DIFICULTY}.csv"
    #
    #     HOUR_EXTRACTED = f"{DIFICULTY_DATA}/out_feat_extraction.csv"
    #     QUARTER_EXTRACTED = f"{DIFICULTY_DATA}/out_feat_extraction_quarters.csv"
    #
    #     # Results figs
    #     RESULTS_FIG = "results/results-data2"
    #     RESULTS_FIG_DIFICULTY = f"{RESULTS_FIG}/{DIFICULTY}"
    #     HOUR_RESULTS = f"{RESULTS_FIG_DIFICULTY}/plot_hours_routines"
    #     QUARTER_RESULTS = f"{RESULTS_FIG_DIFICULTY}/plot_quarters_routines"
    #
    #     # Create paths for results figs
    #     os.makedirs(RESULTS_FIG, exist_ok=True)
    #     os.makedirs(RESULTS_FIG_DIFICULTY, exist_ok=True)
    #     os.makedirs(HOUR_RESULTS, exist_ok=True)
    #     os.makedirs(QUARTER_RESULTS, exist_ok=True)
    #
    #     # Groundtruth figs
    #     GROUNDTRUTH_ROOT = "figs/groundtruth_data2_figs"
    #     GROUNDTRUTH_DIFICULTY = f"{GROUNDTRUTH_ROOT}/{DIFICULTY}"
    #     HOURS_GROUNDTRUTH = f"{GROUNDTRUTH_DIFICULTY}/hours"
    #     QUARTERS_GROUNDTRUTH = f"{GROUNDTRUTH_DIFICULTY}/quarters"
    #
    #     # Create paths for groundtruth figs
    #     os.makedirs(GROUNDTRUTH_ROOT, exist_ok=True)
    #     os.makedirs(GROUNDTRUTH_DIFICULTY, exist_ok=True)
    #     os.makedirs(HOURS_GROUNDTRUTH, exist_ok=True)
    #     os.makedirs(QUARTERS_GROUNDTRUTH, exist_ok=True)
    #
    #     correspondencies = json.load(open(DICTIONARY_FILE))
    #
    #     hour_data = extract_data_grouped_by_hour(DATA_FILE, DICTIONARY_FILE)
    #     quarter_data = extract_data_grouped_by_quarter_hour(DATA_FILE, DICTIONARY_FILE)
    #
    #     hour_data.to_csv(HOUR_EXTRACTED, index=False)
    #     quarter_data.to_csv(QUARTER_EXTRACTED, index=False)
    #
    #     all_rooms = list(correspondencies.keys())
    #     for id_room, room in tqdm(enumerate(all_rooms)):
    #         st = time.time()
    #         path_out_hour = f"{HOUR_RESULTS}/{room}"
    #         path_out_quarter = f"{QUARTER_RESULTS}/{room}"
    #
    #         os.makedirs(path_out_hour, exist_ok=True)
    #         os.makedirs(path_out_quarter, exist_ok=True)
    #
    #         hour_time_series = get_time_series(HOUR_EXTRACTED, room)
    #         quarter_time_series = get_time_series(QUARTER_EXTRACTED, room)
    #
    #         # Obtain rainbow colors
    #         colors = cm.rainbow(np.linspace(0, 1, len(all_rooms)))
    #
    #         plot_hours_groundtruth(time_series=hour_time_series, room=room, barcolors=colors[id_room],
    #                                top_days=15, figsize=(30, 60),
    #                                save_dir=f"{HOURS_GROUNDTRUTH}/{room}.png", show_plot=False)
    #
    #         plot_quarters_groundtruth(time_series=quarter_time_series, room=room, barcolors=colors[id_room],
    #                                   top_days=15, figsize=(50, 60),
    #                                   save_dir=f"{QUARTERS_GROUNDTRUTH}/{room}.png", show_plot=False)
    #
    #         R1, C1, G1 = 5, 5, 20
    #         R2, C2, G2 = 3, 15, 5
    #
    #         if room == "room" and DIFICULTY != "hard":
    #             R1, C1, G1 = 3, 40, 20
    #             R2, C2, G2 = 1, 80, 5
    #
    #         if DIFICULTY == "hard":
    #             R1, C1, G1 = 13, 5, 30
    #             R2, C2, G2 = 5, 10, 5
    #
    #         drgs_hours = DRGS(length_range=(3, 100), R=R1, C=C1, G=G1, epsilon=0.5, L=1, fusion_distance=0.0001)
    #         drgs_quarters = DRGS(length_range=(3, 100), R=R2, C=C2, G=G2, epsilon=0.5, L=1, fusion_distance=0.0001)
    #
    #         drgs_hours.fit(hour_time_series)
    #         drgs_hours.results_per_hour_day(top_days=30, figsize=(30, 120), save_dir=path_out_hour,
    #                                         bars_linewidth=2, show_background_annotations=True,
    #                                         show_plot=False)
    #
    #         tree_hours = drgs_hours.convert_to_cluster_tree()
    #
    #         if drgs_hours.get_results().is_empty():
    #             warnings.warn(f"Empty results for room {room}")
    #
    #         else:
    #             if len(tree_hours.nodes) > 30:
    #                 tree_hours.plot_tree(title="Final node evolution",
    #                                      save_dir=f"{path_out_hour}/final_tree_hours.png",
    #                                      figsize=(27, 27))
    #
    #             elif len(tree_hours.nodes) < 7:
    #                 tree_hours.plot_tree(title="Final node evolution",
    #                                      save_dir=f"{path_out_hour}/final_tree_hours.png",
    #                                      figsize=(7, 7))
    #
    #             else:
    #                 tree_hours.plot_tree(title="Final node evolution",
    #                                      save_dir=f"{path_out_hour}/final_tree_hours.png",
    #                                      figsize=(14, 14))
    #
    #         drgs_quarters.fit(quarter_time_series)
    #         drgs_quarters.results_per_quarter_hour(top_days=30, figsize=(50, 120), save_dir=path_out_quarter,
    #                                                bars_linewidth=2, show_background_annotations=True,
    #                                                show_plot=False)
    #
    #         tree_quarters = drgs_quarters.convert_to_cluster_tree()
    #
    #         if drgs_quarters.get_results().is_empty():
    #             warnings.warn(f"Empty results for room {room}")
    #
    #         else:
    #             if len(tree_quarters.nodes) > 30:
    #                 tree_quarters.plot_tree(title="Final node evolution",
    #                                         save_dir=f"{path_out_quarter}/final_tree_quarters.png",
    #                                         figsize=(27, 27))
    #             elif len(tree_quarters.nodes) < 7:
    #                 tree_quarters.plot_tree(title="Final node evolution",
    #                                         save_dir=f"{path_out_quarter}/final_tree_quarters.png",
    #                                         figsize=(7, 7))
    #
    #             else:
    #                 tree_quarters.plot_tree(title="Final node evolution",
    #                                         save_dir=f"{path_out_quarter}/final_tree_quarters.png",
    #                                         figsize=(14, 14))
    #
    #         print(f"Elapsed time for room {room}: {time.time() - st}")

# Simple fit
# time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
# time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
# target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]
#
# R = 1
# C = 3
# G = 5
# epsilon = 1
#
# drfl = DRFL(3, R,  C,  G,  epsilon)
# drfl.fit(time_series)
#
# detected_routines = drfl.get_results()
#
# drfl.show_results()
# drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
#                        yticks_fontsize=20, figsize=(45, 25),
#                        linewidth_bars=2, xlim=(0, 50))
#
# DRGS = DRGS((3, 8), 2, 3, 5, 1)
# DRGS.fit(time_series)
# DRGS.show_results()
# DRGS.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
#                         yticks_fontsize=20, figsize=(45, 25),
#                         linewidth_bars=2, xlim=(0, 50))

# ts = [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]
# for x in range(4):
#     ts += ts
#
# print(ts)
#
# time_series = pd.Series(np.array(ts))
# # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
# time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
# target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]
#
# _R = [x for x in range(1, 6, 1)]
# _C = [x for x in range(1, 10, 1)]
# _G = [x for x in range(1, 6, 1)]
# _epsilon = [0.5, 1]
# alpha = 0.5
# sigma = 3
#
# parallel_search = ParallelSearchDRFL(n_jobs=10, alpha=alpha, sigma=sigma, param_grid={'_m': 3, '_R': _R, '_C': _C, '_G': _G, '_epsilon': _epsilon})
# parallel_search.fit(time_series, target_centroids=target_centroids)
# results = parallel_search.cv_results()
# best_params = parallel_search.best_params()
# print(results.head())
# print(best_params)
# best_drfl = DRFL(3, best_params["_R"], best_params["_C"], best_params["_G"], best_params["_epsilon"])
# best_drfl.fit(time_series)
#
# detected_routines = best_drfl.get_results()
#
# best_drfl.show_results()
# best_drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
#                        yticks_fontsize=20, figsize=(45, 25),
#                        linewidth_bars=2, xlim=(0,50))

# PARALLEL SEARCH WITH COMPLICATED TIME SERIES

# args = argparser.parse_args()
# df = load_data(args.data_dir)
# correspondencies = obtain_correspondencies(args.dictionary_dir)
# feat_extraction = feature_extraction(df, correspondencies)
# time_series = get_time_series(feat_extraction, "gym")
#
# R_params = [x for x in range(20, 70, 10)]
# C_params = [x for x in range(5, 9)]
# G_params = [x for x in range(20, 100, 10)]
# # L_params = [x for x in range(20, 80, 10)]
#
# target_centroids = [[28, 0, 44, 0, 125, 11, 0], [79, 0, 47, 0, 66, 118, 0]]
# params = list(product(R_params, C_params, G_params, [1], [0]))
# alpha, sigma = 0.6, 4
#
# # Sequential search
# st = time.time()
# result = []
# for R, C, G, epsilon, L in params:
#     drfl = DRFL(7, R, C, G, epsilon, L)
#     drfl.fit(time_series)
#     mean_distance = drfl.estimate_distance(target_centroids, alpha, sigma)
#     result.append({"R": R, "C": C, "G": G, "epsilon": epsilon, "L": L, "mean_distance": mean_distance})
#
# print(f"Elapsed sequential time: {time.time() - st}")
#
# param_grid = {'m': 7, 'R': R_params, 'C': C_params, 'G': G_params, 'epsilon': [1], 'L': [0]}
#
# # Parallel search: comparing time
# st = time.time()
# mDRFL = ParallelSearchDRFL(n_jobs=1, alpha=alpha, sigma=sigma, param_grid=param_grid)
# mDRFL.search_best(time_series, target_centroids=target_centroids)
# print(f"Elapsed time workers=1: {time.time() - st}")
#
# st = time.time()
# mDRFL = ParallelSearchDRFL(n_jobs=5, alpha=alpha, sigma=sigma, param_grid=param_grid)
# mDRFL.search_best(time_series, target_centroids=target_centroids)
# print(f"Elapsed time workers=5: {time.time() - st}")
#
# st = time.time()
# mDRFL = ParallelSearchDRFL(n_jobs=10, alpha=alpha, sigma=sigma, param_grid=param_grid)
# mDRFL.search_best(time_series, target_centroids=target_centroids)
# print(f"Elapsed time workers=10: {time.time() - st}")
#
# st = time.time()
# mDRFL = ParallelSearchDRFL(n_jobs=cpu_count() - 2, alpha=alpha, sigma=sigma,  param_grid=param_grid)
# mDRFL.search_best(time_series, target_centroids=target_centroids)
# print(f"Elapsed time workers={cpu_count() - 2}: {time.time() - st}")
#
# # Results
# results = mDRFL.cv_results()
# best_params = mDRFL.best_params()
# top = results.head()
# print(top)
#
# best_drfl = DRFL(7, best_params["R"], best_params["C"], best_params["G"], best_params["epsilon"], best_params["L"])
# best_drfl.fit(time_series)
# best_drfl.show_results()
# best_drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
#                        yticks_fontsize=20, figsize=(45, 20),
#                        linewidth_bars=2)
