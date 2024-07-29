"""
A module to load data for a specific user from the feature extraction CSV file.
The data is grouped by quarter-hour intervals.
"""

import pandas as pd
from typing import Optional
import os


class DataLoaderUser:
    """
    A class to load data for a specific user from the feature extraction CSV file. The data is grouped by quarter-hour
    intervals.

    Parameters:
    -----------
        - data_dir: `str` The directory where data is stored.
        - user: `str` The user identifier.
        - dificulty: `str` The difficulty level.
        - path: `str` The path to the user's data file.

    Public Methods:
    ----------------
        - ``load_time_series(room: str, select_month: Optional[int] = 3) -> pd.Series:`` Loads the time series data for a specific room and selects a month.
    """

    def __init__(self, data_dir: str, user: str, dificulty: str):
        """
        Initializes the DataLoaderUser with the provided directory, user, and difficulty.

        Parameters:
            data_dir (str): The directory where data is stored.
            user (str): The user identifier.
            dificulty (str): The difficulty level.
        """
        self.__check_params(data_dir, user, dificulty)
        self.data_dir = data_dir
        self.user = user
        self.dificulty = dificulty
        self.path = self.__get_path()

    def __get_path(self) -> str:
        """
        Constructs the file path for the user's data.

        Returns:
            str: The path to the user's data file.
        """
        path = f"{self.data_dir}/{self.user}/{self.dificulty}/out_feat_extraction_quarters.csv"
        return path

    @staticmethod
    def __check_params(data_dir: str, user: str, dificulty: str) -> None:
        """
        Checks the validity of the parameters.

        Parameters:
            data_dir: `str` The directory where data is stored.
            user: `str` The user identifier.
            dificulty: `str` The difficulty level.

        Raises:
            TypeError: If any parameter is of incorrect type.
            FileNotFoundError: If any path does not exist.
        """
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

    def __get_time_series(self, room: str, select_month: Optional[int] = 3) -> pd.Series:
        """
        Extracts the time series data for a specific room.

        Parameters:
            room: `str` The room identifier.
            select_month: `Optional[int]` The month to filter data by. Defaults to 3.

        Returns:
            pd.Series: The time series data for the specified room.

        Raises:
            TypeError: If the room is not a string or select_month is not an integer or None.
        """

        valid_months = list(range(1, 13))

        # Check the validity of the parameters
        if not isinstance(room, str):
            raise TypeError(f"Room must be a string. Got {type(room).__name__}")

        if not isinstance(select_month, int) and select_month is not None:
            raise TypeError(f"Select month must be an integer or None. Got {type(select_month).__name__}")

        if select_month is not None and select_month not in valid_months:
            raise ValueError(f"Invalid month. Must be between 1 and 12. Got {select_month}")

        # Read the feature extraction CSV file
        feat_extraction = pd.read_csv(self.path)

        # Combine Year, Month, Day, and Hour into a single DateTime column
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])

        # Adjust DateTime column to include the quarter-hour information
        for row in range(feat_extraction.shape[0]):
            feat_extraction.loc[row, "Date"] = feat_extraction.loc[row, "Date"] + pd.Timedelta(
                minutes=feat_extraction.loc[row, "Quarter"] * 15)

        # Set DateTime as the index
        feat_extraction.set_index("Date", inplace=True)

        # Filter data by the selected month if provided
        if select_month is not None:
            feat_extraction = feat_extraction[feat_extraction["Month"] == int(select_month)]

        # Extract the time series for the specified room
        room_time_series = feat_extraction[f"N_{room}"]

        return room_time_series

    def load_time_series(self, room: str, select_month: Optional[int] = 3) -> pd.Series:
        """
        Public method to load the time series for a specific room.

        Parameters:
            room (str): The room identifier.
            select_month (Optional[int]): The month to filter data by. Defaults to 3. Valid values are 1-12 or None.

        Returns:
            pd.Series: The time series data for the specified room.
        """

        return self.__get_time_series(room=room, select_month=select_month)
