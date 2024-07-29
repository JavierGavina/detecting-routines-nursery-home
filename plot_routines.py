import warnings
import numpy as np

import os
import yaml

from src.DRFL import DRGS
from utils.dataloader import DataLoaderUser
from utils.visualization import save_results_plot, plot_quarters_groundtruth


def get_path_results(path: str, user: str, dificulty: str) -> str:
    return f"{path}/{user}/{dificulty}"


def setup() -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def get_params(config: dict, room: str) -> dict:
    return config["params"]["Room"] if room == "Room" else config["params"]["other"]


if __name__ == "__main__":
    config = setup()
    os.makedirs(config["results_dir"], exist_ok=True)

    routine_visualization_dir = os.path.join(config["results_dir"], "routine_visualization")
    data_visualization_dir = os.path.join(config["results_dir"], "data_visualization")

    os.makedirs(routine_visualization_dir, exist_ok=True)
    os.makedirs(data_visualization_dir, exist_ok=True)

    for user in config["users"]:

        os.makedirs(os.path.join(data_visualization_dir, user), exist_ok=True)
        os.makedirs(os.path.join(routine_visualization_dir, user), exist_ok=True)

        for dificulty in config["dificulties"]:
            room_visualization = get_path_results(path=data_visualization_dir, user=user, dificulty=dificulty)
            room_routines = get_path_results(path=routine_visualization_dir, user=user, dificulty=dificulty)

            os.makedirs(room_visualization, exist_ok=True)
            os.makedirs(room_routines, exist_ok=True)

            data_loader = DataLoaderUser(data_dir=config["data_dir"], user=user, dificulty=dificulty)

            for room in config["rooms"]:
                time_series = data_loader.load_time_series(room=room)

                if np.sum(time_series) == 0:
                    warnings.warn(f"Room {room} has no data for user {user} and dificulty {dificulty}")
                    continue

                xlim = ("09:30", "20:00") if room != "Room" else None

                plot_quarters_groundtruth(time_series=time_series,
                                          barcolors=config["colors"][room],
                                          top_days=7, figsize=(15, 23), linewidth=2,
                                          xlim=xlim, show_grid=False,
                                          save_dir=f"{room_visualization}/{room}.pdf", show_plot=False)

                params = get_params(config=config, room=room)
                drgs = DRGS(**params)
                drgs.fit(time_series=time_series, verbose=False)
                save_results_plot(drgs_fitted=drgs, room_routines=room_routines, user=user, dificulty=dificulty,
                                  room=room)
