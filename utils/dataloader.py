import pandas as pd
from typing import Optional
import os


class DataLoaderUser:
    def __init__(self, data_dir: str, user: str, dificulty: str):
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
