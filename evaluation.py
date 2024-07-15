from typing import Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib import gridspec

from src.DRFL import DRGS
from src.structures import Cluster, HierarchyRoutine
import datetime

import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
intervalos_temporales = [("10:15", "11:15"), ("11:30", "12:45"), ("13:30", "14:45"), ("15:00", "16:15"),
                         ("16:45", "17:45"), ("18:00", "19:15")]


def get_time_series(path_to_feat_extraction: str, room: str, select_month: Optional[str] = None) -> pd.Series:
    """
    Get the time series of the room

    Parameters:
        path_to_feat_extraction: `str` path to the feature extraction
        room: `str` room to get the time series
        select_month: `Optional[str]` month to select

    Returns:
        `pd.Series` time series of the room
    """

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


def filter_dates(dates: list[datetime.date], weekday: str, interval: tuple[int, int]) -> list[datetime.date]:
    """
    Filter the dates that are in the weekday and interval specified

    Parameters:
        dates: `list[datetime.date]` list of dates
        weekday: `str` weekday
        interval: `tuple[str, str]` interval of time

    Returns:
        `list[datetime.date]` list of dates that are in the weekday and interval specified

    """
    start, end = interval
    # start = int(start.split(":")[0]) * 60 + int(start.split(":")[1])
    # end = int(end.split(":")[0]) * 60 + int(end.split(":")[1])

    return [date for date in dates if
            date.weekday() == weekdays.index(weekday) and \
            start <= date.hour * 60 + date.minute <= end]


def instances_to_minutes_on_weekday(*,
                                    instances: list[datetime.date],
                                    duration_interval: int,
                                    N_days_per_week: int,
                                    K: int = 15,
                                    m: int = 3) -> float:
    """
    Get the number of minutes that the instances are on the weekday and interval specified

    Parameters:
        instances: `list[datetime.date]` list of instances
        weekday: `str` weekday
        interval: `tuple[str, str]` interval of time
        duration_interval: `int` duration of the interval
        K: `int` number of minutes per time interval
        m: `int` hierarchy level

    Returns:
        `int` number of minutes that the instances are on the weekday and interval specified
    """

    # separate the instance per day
    instances_per_day = {}
    for instance in instances:
        day = instance.day
        if day not in instances_per_day:
            instances_per_day[day] = []
        instances_per_day[day].append(instance)

    # get n consecutive instances per day
    N_cons_per_day = {}
    for day in instances_per_day:
        instances_of_day = instances_per_day[day]
        # diference between the first and last instance
        diff = instances_of_day[-1] - instances_of_day[0]
        Ncons = diff.total_seconds() / (K * 60)
        N_cons_per_day[day] = Ncons

    minutes_per_day = {}
    for day in instances_per_day.keys():
        minutes_per_day[day] = min(K * m + N_cons_per_day[day] * K, duration_interval)

    return sum(minutes_per_day.values()) / (N_days_per_week * duration_interval)


#
# def summarize_cluster(cluster: Cluster) -> pd.DataFrame:
#     """
#     Parameters:
#         cluster: `Cluster` cluster to summarize
#
#     Returns:
#         pd.DataFrame: summary of the cluster
#     """
#
#     cluster_info = pd.DataFrame(columns=["Weekday", "Interval", "Frequency"])
#     dates = cluster.get_dates()
#     for weekday in range(7):
#         for interval in range(6):
#             dates_query = filter_dates(dates, weekdays[weekday], intervalos_temporales[interval])
#             cluster_info.loc[len(cluster_info)] = [weekdays[weekday], intervalos_temporales[interval], len(dates_query)]
#
#     return cluster_info
#
#
# def get_summary_from_graph(hierarchy_routine: HierarchyRoutine, group_by_hierarhcy: bool = False) -> dict[
#     str, pd.DataFrame]:
#     """
#     Get the summary from the graph of the hierarchy routine
#
#     Parameters:
#         hierarchy_routine: `HierarchyRoutine` hierarchy routine estimated
#         group_by_hierarhcy: `bool` if True, a frequency table is obtained per hierarchy. Otherwise, a frequency table is per each cluster
#
#     Returns:
#         `dict[str, pd.DataFrame]` dictionary with the summary of frequency tables
#
#     """
#     relative_table_per_cluster = dict()
#     for hierarchy in range(min(hierarchy_routine.keys), max(hierarchy_routine.keys) + 1):
#         routine_m = hierarchy_routine[hierarchy]
#         for id_cluster, cluster in enumerate(routine_m):
#             relative_table_per_cluster[f"{hierarchy}-{id_cluster + 1}"] = summarize_cluster(cluster)
#
#     if not group_by_hierarhcy:
#         return relative_table_per_cluster
#
#     relative_table_per_hierarchy = dict()
#     for hierarchy in range(min(hierarchy_routine.keys), max(hierarchy_routine.keys) + 1):
#         # Sumar tabla frecuencia elemento a elemento
#         sum_frequencies = []
#         for key in relative_table_per_cluster.keys():
#             if key.split("-")[0] == str(hierarchy):
#                 if len(sum_frequencies) == 0:  # Si es la primera vez
#                     sum_frequencies = relative_table_per_cluster[key]
#                 else:
#                     sum_frequencies["Frequency"] += relative_table_per_cluster[key]["Frequency"]
#
#         relative_table_per_hierarchy[str(hierarchy)] = sum_frequencies
#
#     return relative_table_per_hierarchy
#
#
# def transform_to_table(summary: pd.DataFrame) -> pd.DataFrame:
#     """
#     Transform the summary to the frequency table
#
#     Parameters:
#         summary: `pd.DataFrame` summary of the clusters
#
#     Returns:
#         `pd.DataFrame` frequency table
#     """
#
#     table = summary.pivot_table(index="Weekday", columns="Interval", values="Frequency").reset_index()
#     table["Weekday"] = table["Weekday"].apply(lambda x: weekdays.index(x))
#     table.sort_values(by="Weekday", inplace=True)
#     table["Weekday"] = table["Weekday"].apply(lambda x: weekdays[x])
#
#     return table
#
#
# def get_distance_matrix(summary: pd.DataFrame) -> np.ndarray:
#     """
#     Get the distance matrix from the frequency table
#
#     Parameters:
#         summary: `pd.DataFrame` summary of the clusters
#
#     Returns:
#         `np.ndarray` matrix with the distance information
#     """
#
#     table = transform_to_table(summary).iloc[:, 1:].values
#     distance_matrix = pdist(table, metric="euclidean")
#
#     return distance_matrix
#
#
# def get_linkage_matrix(distance_matrix: np.ndarray) -> np.ndarray:
#     """
#     Get the linkage matrix using the distance matrix and the ward method
#
#     Parameters:
#         distance_matrix: `np.ndarray` euclidean matrix with the distance information
#
#     Returns:
#         `np.ndarray` matrix with the linkage information
#     """
#     return hc.linkage(distance_matrix, method="ward")
#
#
# def plot_dendrogram(linkage_matrix: np.ndarray,
#                     labels: np.ndarray,
#                     figsize: tuple[int, int] = (10, 5),
#                     title: str = "Dendograma",
#                     save_dir: str = None,
#                     show_plot: bool = True):
#     """
#     Plot the dendrogram
#
#     Parameters:
#         linkage_matrix: `np.ndarray` matrix with the linkage information
#         labels: `np.ndarray` list with the labels
#         figsize: `tuple[int, int]` size of the figure
#         save_dir: `str` path to save the figure
#         show_plot: `bool` show the plot
#     """
#
#     plt.figure(figsize=figsize)
#     hc.dendrogram(linkage_matrix, labels=labels)
#     plt.ylim(0, 54)
#     plt.grid(True)
#     plt.title(title)
#     plt.axhline(y=10, color='r', linestyle='--')
#     plt.ylabel("Distancia Ward")
#     if save_dir is not None:
#         format = save_dir.split(".")[-1]
#         plt.savefig(save_dir, format=format)
#     if show_plot:
#         plt.show()

def setup() -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def get_params(config: dict, room: str) -> dict:
    return config["params"]["Room"] if room == "Room" else config["params"]["other"]


if __name__ == "__main__":
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    config = setup()
    ROOM = "Terrace"
    DIFICULTY = "hard"
    path_to_feat_extraction = f"data/9FE9/{DIFICULTY}/out_feat_extraction_quarters.csv"
    params_drgs = get_params(config, ROOM)
    intervals_user = config["intervals_of_interest"]["9FE9"]
    N_days_per_week = config["N_weekdays_on_month"]
    time_series = get_time_series(path_to_feat_extraction, ROOM, select_month="3")
    drgs = DRGS(**params_drgs)
    drgs.fit(time_series)
    hierarchy_routine = drgs.get_results()
    for key in hierarchy_routine.keys:
        print(f"Hierarchy: {key}")
        for id_cluster, cluster in enumerate(hierarchy_routine[key]):
            for weekday in weekdays:
                for interval in intervals_user:
                    dates = filter_dates(cluster.get_dates(), weekday, (interval["start"], interval["end"]))
                    relative_frequency = instances_to_minutes_on_weekday(instances=dates, duration_interval=interval["duration_time"],
                                                                         N_days_per_week=N_days_per_week[weekday], K=15, m=key)
                    print(f"Cluster: {key}-{id_cluster+1}; {weekday}; {interval}; Relative frequency: {relative_frequency}")
            print("\n\n\n")

