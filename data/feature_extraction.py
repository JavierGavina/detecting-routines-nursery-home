"""
This script is used to extract the features from the data.
The features are extracted by grouping the data in quarter
hours and counting the number of occurrences of each room
in each quarter-hour. The data is then saved in a csv file as out_feat_extraction_quartes.csv.
This file will be used to obtain the time series of each room from a user and difficulty.
"""
import pandas as pd
import json
import numpy as np
import yaml


def setup() -> dict:
    """
    Loads the configuration settings from a YAML file.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    with open("../config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def extract_data_grouped_by_quarter_hour(path_to_data: str, path_to_dictionary: str) -> pd.DataFrame:
    """
    Extracts data grouped by quarter-hour intervals and counts the occurrences of each room.

    Parameters:
        path_to_data: `str` The path to the CSV file containing the data.
        path_to_dictionary: `str` The path to the JSON file containing room correspondencies.

    Returns:
        pd.DataFrame: A DataFrame with the extracted features.
    """
    # Read the data from the CSV file
    data = pd.read_csv(path_to_data, skiprows=1, header=None)

    # Read the room correspondencies from the JSON file
    correspondencies = json.load(open(path_to_dictionary))

    # Rename the columns of the data
    data.columns = ["Year", "Month", "Day"] + [f"Sequence_{x}" for x in range(1439)]

    # Create the output columns
    out_columns = ["Year", "Month", "Day", "Hour", "Quarter"] + [f"N_{x}" for x in correspondencies.keys()]
    new_df = pd.DataFrame(columns=out_columns)

    for row in range(data.shape[0]):
        for hour in range(24):
            for quarter in range(4):

                # Extract the sequence of rooms for the quarter-hour interval
                sequence = data.iloc[row, 3 + hour * 60 + quarter * 15:3 + hour * 60 + (quarter + 1) * 15].replace(
                    np.nan, 0).values

                # Count the occurrences of each room
                frequency_rooms = [np.sum(sequence == i) for i in correspondencies.values()]

                # Add the data to the output DataFrame
                year, month, day = data.iloc[row, 0], data.iloc[row, 1], data.iloc[row, 2]
                new_df.loc[len(new_df)] = [year, month, day, hour, quarter] + frequency_rooms

    return new_df


if __name__ == "__main__":
    config = setup()
    for user in config["users"]:
        dictionary_file = f"{user}/metadata/dictionary_rooms.json"
        for dificulty in config["dificulties"]:
            data_file = f"{user}/{dificulty}/activities-simulation.csv"
            out_extraction_path = f"{user}/{dificulty}/out_feat_extraction_quarters.csv"
            quarted_data = extract_data_grouped_by_quarter_hour(data_file, dictionary_file)
            quarted_data.to_csv(out_extraction_path, index=False)
