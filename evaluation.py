from typing import Union

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from src.DRFL import DRGS
from src.structures import HierarchyRoutine, Routines, Cluster
import datetime
from itertools import combinations
from utils.dataloader import DataLoaderUser


def setup() -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


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

    return [date for date in dates if
            date.weekday() == weekdays.index(weekday) and \
            start <= date.hour * 60 + date.minute <= end]


def get_relative_frequency(*,
                           instances: list[datetime.date],
                           duration_interval: int,
                           N_days_per_week: int,
                           K: int = 15,
                           m: int = 3) -> float:
    """
    Get the number of minutes that the instances are on the weekday and interval specified

    Parameters:
        instances: `list[datetime.date]` list of instances
        duration_interval: `int` duration of the interval
        N_days_per_week: `int` number of days the weekday appears on the month
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


def get_params(room: str) -> dict:
    """
    Get the parameters of the room or other location

    Parameters:
        room: `str` room name

    Returns:
        `dict` parameters of the room
    """

    return config["params"]["Room"] if room == "Room" else config["params"]["other"]


def convert_from_minutes_to_hour(minutes: int) -> str:
    """
    Convert the minutes to hour

    Parameters:
        minutes: `int` minutes to convert

    Returns:
        `str` hour in format HH:MM
    """

    hour = minutes // 60
    minutes = minutes % 60

    return f"{hour}:{minutes:02d}"


def summarize_cluster(user: str, location: str, cluster: Cluster) -> pd.DataFrame:
    intervals_user = config["intervals_of_interest"][user]
    N_days_per_week = config["N_weekdays_on_month"]
    cluster_info = pd.DataFrame(columns=["User", "Location", "Weekday", "Start", "End", "RelativeFrequency"])
    m_param = cluster.length_cluster_subsequences
    for weekday in weekdays:
        for interval in intervals_user:
            dates = filter_dates(cluster.get_dates(), weekday, (interval["start"], interval["end"]))
            relative_frequency = get_relative_frequency(instances=dates,
                                                        duration_interval=interval["duration_time"],
                                                        N_days_per_week=N_days_per_week[weekday], K=15,
                                                        m=m_param)
            start = convert_from_minutes_to_hour(interval["start"])
            end = convert_from_minutes_to_hour(interval["end"])
            cluster_info.loc[len(cluster_info)] = [user, location, weekday, start, end, relative_frequency]

    return cluster_info


def inclusion_exclusion(list_of_probabilities: Union[list[float], pd.Series]) -> float:
    """
    Inclusion-exclusion principle, the probability of the union of n events

    You can see the formula in the following link:
    https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle

    Parameters:
        list_of_probabilities: `list[float]` or `pd.Series` list of probabilities

    Returns:
        `float` probability of the union of n events

    Examples:
    >>> inclusion_exclusion([0.2, 0.1, 0.33333])
    0.520

    >>> inclusion_exclusion([0.4, 0.4, 0.2])
    0.712
    """

    n = len(list_of_probabilities)
    result = 0
    for i in range(1, n + 1):
        for comb in combinations(list_of_probabilities, i):
            result += (-1) ** (i + 1) * np.prod(comb)

    return np.round(result, decimals=4)


def union_n_columns(df: pd.DataFrame) -> pd.Series:
    """
    Apply the inclusion-exclusion principle to the columns of the DataFrame for each row

    Parameters:
        df: `pd.DataFrame` DataFrame to apply the union of the probabilities

    Returns:
        `pd.Series` union of the probabilities of the columns from the DataFrame for each row
    """

    return df.apply(lambda x: inclusion_exclusion(x), axis=1)


def fusion_all_clusters(routine: Routines) -> Cluster:
    fusioned_cluster = routine[0]
    for id_cluster in range(1, len(routine)):
        fusioned_cluster += routine[id_cluster]

    return fusioned_cluster


def hierarchy_summarization(*, hierarchy_routine: HierarchyRoutine, key: int, user: str, location: str) -> pd.DataFrame:
    """
    Get the summary of the hierarchy routine for a specific key
    applying the union of the probabilities for each cluster of
    a specified hierarchy using the inclusion-exclusion principle

    Parameters:
        hierarchy_routine: `HierarchyRoutine` hierarchy routine estimated
        key: `int` hierarchy or length parameter of the hierarchy routine
        user: `str` username
        location: `str` location of the user

    Returns:
        `pd.DataFrame` Table of relative frequencies of the hierarchy routine for a hierarchy
    """

    relative_table_per_cluster = dict()
    routine_m = hierarchy_routine[key]
    if len(routine_m) <= 10:
        result = summarize_cluster(user, location, routine_m[0])
        for id_cluster, cluster in enumerate(routine_m):
            relative_table_per_cluster[f"{id_cluster + 1}"] = summarize_cluster(user, location, cluster)["RelativeFrequency"]

        relative_table = pd.DataFrame(relative_table_per_cluster)
        joined_columns = union_n_columns(relative_table)
        result["RelativeFrequency"] = joined_columns

    else:
        print(f"Fusioning all clusters in user: {user}, location: {location}, for hierarchy: {key}")
        fusioned_cluster = fusion_all_clusters(routine_m)
        result = summarize_cluster(user, location, fusioned_cluster)

    return result


if __name__ == "__main__":
    config = setup()
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for dificulty in config["dificulties"]:
        user_groundtruth = pd.DataFrame(columns=["User", "Location", "WeekDay", "Start", "End", "RelativeFrequency"])
        for user in config["users"]:
            data_loader = DataLoaderUser(data_dir=config["data_dir"], user=user, dificulty=dificulty)
            for location in tqdm(config["rooms"]):
                params_drgs = get_params(location)
                time_series = data_loader.load_time_series(room=location)

                if np.sum(time_series) == 0:
                    continue

                drgs = DRGS(**params_drgs)
                drgs.fit(time_series, verbose=False)
                hierarchy_routine = drgs.get_results()

                if hierarchy_routine.is_empty():
                    continue

                relative_frequency = hierarchy_summarization(hierarchy_routine=hierarchy_routine, key=3, user=user, location=location)
                user_groundtruth = pd.concat([user_groundtruth, relative_frequency])

        user_groundtruth.to_csv(f"results/{dificulty}_frequency_table.csv", index=False)
