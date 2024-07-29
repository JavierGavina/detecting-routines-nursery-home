import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import Union, Dict, Tuple
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, f1_score, precision_score, \
    recall_score
import yaml
from sklearn.preprocessing import LabelBinarizer

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def setup() -> dict:
    """
    Loads the configuration settings from a YAML file.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def get_groundtruth_locations(groundtruth: pd.DataFrame) -> dict:
    """
    Get unique locations detected for each user from the groundtruth data.

    Parameters:
        groundtruth (pd.DataFrame): Groundtruth DataFrame.

    Returns:
        dict: Dictionary with users as keys and lists of locations as values.
    """
    locations_detected_per_user = {}
    for user in groundtruth["User"].unique():
        locations_detected_per_user[user] = groundtruth[groundtruth["User"] == user]["Location"].unique().tolist()
    return locations_detected_per_user


def extend_estimated_table(estimated: pd.DataFrame, groundtruth: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the estimated table with locations from the groundtruth.

    Parameters:
        estimated (pd.DataFrame): Estimated DataFrame.
        groundtruth (pd.DataFrame): Groundtruth DataFrame.

    Returns:
        pd.DataFrame: Extended DataFrame.
    """
    locations_detected_per_user = get_groundtruth_locations(groundtruth)
    result = pd.DataFrame(columns=estimated.columns)
    for user in estimated["User"].unique():
        groundtruth_locations = locations_detected_per_user[user]
        user_query = estimated.query(f"User=='{user}'")
        starts = user_query["Start"].unique()
        ends = user_query["End"].unique()
        for location in groundtruth_locations:
            if location not in user_query["Location"].unique():
                for weekday in WEEKDAYS:
                    for start, end in zip(starts, ends):
                        result.loc[len(result)] = [user, location, weekday, start, end, 0.0]
            else:
                location_query = user_query.query(f"Location=='{location}'")
                for row, value in location_query.iterrows():
                    result.loc[len(result)] = value.tolist()
    return result


def get_groundtruth_from_path(path: str) -> pd.DataFrame:
    """
    Load and prepare the groundtruth data from a file.

    Parameters:
        path (str): Path to the groundtruth file.

    Returns:
        pd.DataFrame: Prepared groundtruth DataFrame.
    """
    groundtruth = pd.read_csv(path, sep=";").iloc[:-1, :]
    groundtruth["WeekDay"] = pd.Categorical(groundtruth["WeekDay"], categories=WEEKDAYS, ordered=True)
    return groundtruth.sort_values(["User", "Location", "WeekDay", "Start", "End"])


def get_estimated_from_path(path_to_estimated: str, path_to_groundtruth: str) -> pd.DataFrame:
    """
    Load and extend the estimated data from a file.

    Parameters:
        path_to_estimated (str): Path to the estimated data file.
        path_to_groundtruth (str): Path to the groundtruth data file.

    Returns:
        pd.DataFrame: Extended and sorted estimated DataFrame.
    """
    groundtruth = get_groundtruth_from_path(path_to_groundtruth)
    estimated = pd.read_csv(path_to_estimated, sep=",")
    extended = extend_estimated_table(estimated, groundtruth)
    extended["WeekDay"] = pd.Categorical(extended["WeekDay"], categories=WEEKDAYS, ordered=True)
    return extended.sort_values(["User", "Location", "WeekDay", "Start", "End"])


def classificate(data: pd.DataFrame, labelize: bool = False, threshold=0.5) -> pd.DataFrame:
    """
    Classify the data based on relative frequency and a threshold.

    Parameters:
        data (pd.DataFrame): Data to classify.
        labelize (bool): Whether to labelize the data. Default is False.
        threshold (float): Threshold for classification. Default is 0.5.

    Returns:
        pd.DataFrame: Classified DataFrame.
    """
    unique_intervals = data[['Start', 'End']].drop_duplicates()
    user = data["User"].values[0]
    result = pd.DataFrame(columns=["User", "WeekDay", "Start", "End", "Location"])
    for weekday in WEEKDAYS:
        for start, end in zip(unique_intervals["Start"], unique_intervals["End"]):
            query = data.query(f"Start=='{start}' and End=='{end}' and WeekDay=='{weekday}'")
            if query.RelativeFrequency.max() > threshold:
                estimated_location = query.iloc[np.argmax(query.RelativeFrequency)]["Location"]
            else:
                estimated_location = "missclassified"
            if labelize:
                result.loc[len(result)] = [user, weekday, start, end, estimated_location]
            else:
                result.loc[len(result)] = [user, weekday, start, end, config["rooms"].index(estimated_location)]
    return result


def labelize_and_probabilities(data: pd.DataFrame, groundtruth: pd.DataFrame) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Labelize and calculate probabilities for the data.

    Parameters:
        data (pd.DataFrame): Data to labelize and calculate probabilities for.
        groundtruth (pd.DataFrame): Groundtruth data.

    Returns:
        tuple: Tuple containing labelized groundtruth, labelized data, and probabilities.
    """
    classes = np.union1d(data["Location"].unique(), groundtruth["Location"].unique())
    unique_intervals = data[['Start', 'End']].drop_duplicates()
    labels = []
    probabilities = []
    labels_groundtruth = []
    exist_some_missclassified = False
    for weekday in WEEKDAYS:
        for start, end in zip(unique_intervals["Start"], unique_intervals["End"]):
            query = data.query(f"Start=='{start}' and End=='{end}' and WeekDay=='{weekday}'")
            probabilities.append(query.RelativeFrequency.tolist())
            if query.RelativeFrequency.max() > 0.5:
                labels.append(query.iloc[np.argmax(query.RelativeFrequency)]["Location"])
            else:
                exist_some_missclassified = True
                labels.append("missclassified")
                probabilities[-1] = [0.0] * len(classes) + [1.0]

            query_groundtruth = groundtruth.query(f"Start=='{start}' and End=='{end}' and WeekDay=='{weekday}'")
            labels_groundtruth.append(
                query_groundtruth["Location"].iloc[np.argmax(query_groundtruth.RelativeFrequency)])

    if exist_some_missclassified:
        len_list = [len(prob) for prob in probabilities]
        for i in range(len(probabilities)):
            if len(probabilities[i]) < max(len_list):
                probabilities[i] = probabilities[i] + [0.0]

    classes = np.array(classes.tolist() + ["missclassified"])
    lb = LabelBinarizer()
    lb.fit(classes)

    return lb.transform(labels_groundtruth), lb.transform(labels), np.array(probabilities)


def get_roc_auc(labels_groundtruth: np.ndarray, user_probabilities: np.ndarray, classes: np.ndarray) -> tuple[dict, dict, dict]:
    """
    Compute ROC and AUC for the given labels and probabilities.

    Parameters:
        labels_groundtruth (np.ndarray): Groundtruth labels.
        user_probabilities (np.ndarray): User probabilities.
        classes (np.ndarray): Class labels.

    Returns:
        tuple: False positive rates, true positive rates, and ROC AUC.
    """
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_groundtruth[:, i], user_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_groundtruth[:, :n_classes].ravel(),
                                              user_probabilities[:, :n_classes].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def plot_roc(user: str, dificulty: str):
    """
    Plot the ROC curve for a given user and difficulty.

    Parameters:
        user: `str` User identifier.
        dificulty: `str` Difficulty level.
    """
    groundtruth_path = os.path.join(config["data_dir"], "groundtruth.csv")

    path_estimated = {
        "easy": os.path.join(config["results_dir"], "easy_frequency_table.csv"),
        "medium": os.path.join(config["results_dir"], "medium_frequency_table.csv"),
        "hard": os.path.join(config["results_dir"], "hard_frequency_table.csv")
    }

    data = get_estimated_from_path(path_estimated[dificulty], groundtruth_path)
    estimated_user = data.query(f"User=='{user}'")
    groundtruth_user = get_groundtruth_from_path(groundtruth_path).query(f"User=='{user}'")
    classes = np.union1d(estimated_user["Location"].unique(), groundtruth_user["Location"].unique())
    labels_groundtruth, user_labels, user_probabilities = labelize_and_probabilities(estimated_user, groundtruth_user)

    fpr, tpr, roc_auc = get_roc_auc(labels_groundtruth, user_probabilities, classes)

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for idx, c in enumerate(classes):
        plt.plot(fpr[idx], tpr[idx], color=np.array(config["colors"][c]) / 255, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(c, roc_auc[idx]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {user}')
    plt.legend(loc="lower right")
    plt.show()


def get_binary_statistics(ytrue: np.ndarray, ypred: np.ndarray) -> tuple[float, float, float]:
    """
    Get binary classification statistics.

    Parameters:
        ytrue (np.ndarray): Groundtruth labels.
        ypred (np.ndarray): Predicted labels.

    Returns:
        tuple: F1-score, precision, recall, and AUC.
    """

    f1_room = np.round(f1_score(ytrue, ypred, zero_division=0), 3)
    precision_room = np.round(precision_score(ytrue, ypred, zero_division=0), 3)
    recall_room = np.round(recall_score(ytrue, ypred, zero_division=0), 3)

    return f1_room, precision_room, recall_room


def get_multiclass_statistics(groundtruth: np.ndarray, estimated: np.ndarray) -> tuple[float, float, float]:
    """
    Get multiclass classification statistics.

    Parameters:
        groundtruth: `np.ndarray` Groundtruth labels.
        estimated: `np.ndarray` Predicted labels.

    Returns:
        tuple: F1-score, precision and recall
    """

    f1_macro = np.round(f1_score(y_true=groundtruth, y_pred=estimated, average="macro", zero_division=0), decimals=3)
    f1_weighted = np.round(f1_score(y_true=groundtruth, y_pred=estimated, average="weighted", zero_division=0), decimals=3)
    f1_micro = np.round(f1_score(y_true=groundtruth, y_pred=estimated, average="micro", zero_division=0), decimals=3)

    return f1_macro, f1_weighted, f1_micro


if __name__ == "__main__":
    config = setup()
    PATH_METRICS = os.path.join(config["results_dir"], "metrics")
    PATH_ROC = os.path.join(PATH_METRICS, "roc")
    PATH_CONFUSION_MATRIX = os.path.join(PATH_METRICS, "confusion_matrix")
    PATH_STATISTICS = os.path.join(PATH_METRICS, "statistics")

    os.makedirs(PATH_METRICS, exist_ok=True)
    os.makedirs(PATH_ROC, exist_ok=True)
    os.makedirs(PATH_CONFUSION_MATRIX, exist_ok=True)
    os.makedirs(PATH_STATISTICS, exist_ok=True)

    PATH_GROUNDTRUTH = os.path.join(config["data_dir"], "groundtruth.csv")
    PATH_EASY = os.path.join(config["results_dir"], "easy_frequency_table.csv")
    PATH_MEDIUM = os.path.join(config["results_dir"], "medium_frequency_table.csv")
    PATH_HARD = os.path.join(config["results_dir"], "hard_frequency_table.csv")

    user_statistics = pd.DataFrame(
        columns=["User", "Dificulty", "F1-Macro", "F1-Weighted", "F1-micro", "Macro-AUC", "Micro-AUC"])
    rooms_statistics = pd.DataFrame(columns=["Dificulty", "User", "Room", "F1-Score", "Precision", "Recall", "AUC"])

    # Get statistics for each user and dificulty level and ROC curves
    for dificulty in ["easy", "medium", "hard"]:

        groundtruth = get_groundtruth_from_path(PATH_GROUNDTRUTH)
        dificulty_data = get_estimated_from_path(eval(f"PATH_{dificulty.upper()}"), PATH_GROUNDTRUTH)
        plt.subplots(2, 3, figsize=(20, 10))

        for user in config["users"]:
            estimated_user = dificulty_data.query(f"User=='{user}'")
            groundtruth_user = groundtruth.query(f"User=='{user}'")
            classes = np.union1d(estimated_user["Location"].unique(), groundtruth_user["Location"].unique())
            labels_groundtruth, user_labels, user_probabilities = labelize_and_probabilities(estimated_user, groundtruth_user)
            fpr, tpr, roc_auc = get_roc_auc(labels_groundtruth, user_probabilities, classes)
            if len(user_labels) > len(classes):
                classes = np.append(classes, "missclassified")

            for idx, c in enumerate(classes[:-1]):
                f1, precision, recall = get_binary_statistics(ytrue=labels_groundtruth[:, idx],
                                                              ypred=user_labels[:, idx])
                auc_room = np.round(roc_auc[idx], 3)
                rooms_statistics.loc[len(rooms_statistics)] = [dificulty, user, c, f1, precision, recall, auc_room]

            macro, weighted, micro = get_multiclass_statistics(groundtruth=labels_groundtruth, estimated=user_labels)
            macro_auc = np.round(roc_auc["macro"], decimals=3)
            micro_auc = np.round(roc_auc["micro"], decimals=3)
            user_statistics.loc[len(user_statistics)] = [user, dificulty, macro, weighted, micro, macro_auc, micro_auc]

            # Plot all ROC curves
            plt.subplot(2, 3, config["users"].index(user) + 1)
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            for idx, c in enumerate([c for c in classes if c != "missclassified"]):
                plt.plot(fpr[idx], tpr[idx], color=np.array(config["colors"][c]) / 255, lw=2,
                         label='ROC curve of {0} (area = {1:0.2f})'
                               ''.format(c, roc_auc[idx]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC: {user}')
            plt.legend(loc="lower right")
        plt.savefig(os.path.join(PATH_ROC, f"roc_{dificulty}.png"))
        plt.show()

    user_statistics.to_csv(os.path.join(PATH_STATISTICS, "user_statistics.csv"), index=False)
    rooms_statistics.to_csv(os.path.join(PATH_STATISTICS, "rooms_statistics.csv"), index=False)

    # Plot confusion matrix for each user and dificulty level
    for dificulty in config["dificulties"]:

        # Get the groundtruth and estimated data
        groundtruth = get_groundtruth_from_path(PATH_GROUNDTRUTH)
        dificulty_data = get_estimated_from_path(eval(f"PATH_{dificulty.upper()}"), PATH_GROUNDTRUTH)
        plt.subplots(2, 3, figsize=(20, 10))
        for user in config["users"]:
            # Get the estimated and groundtruth data for the user
            estimated_user = dificulty_data.query(f"User=='{user}'")
            groundtruth_user = groundtruth.query(f"User=='{user}'")
            classes = np.union1d(estimated_user["Location"].unique(), groundtruth_user["Location"].unique())

            # Labelize and calculate probabilities
            labels_groundtruth, user_labels, user_probabilities = labelize_and_probabilities(estimated_user,
                                                                                             groundtruth_user)
            # Add the missclassified class if it exists
            if len(user_labels) > len(classes):
                classes = np.append(classes, "missclassified")

            # Compute the confusion matrix
            cm = confusion_matrix(y_true=classes[labels_groundtruth.argmax(axis=1)],
                                  y_pred=classes[user_labels.argmax(axis=1)])

            # Plot the confusion matrix
            plt.subplot(2, 3, config["users"].index(user) + 1)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(user)
            cb = plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=30)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            cb.remove()
        plt.savefig(os.path.join(PATH_CONFUSION_MATRIX, f"confusion_matrix_{dificulty}.png"))
        plt.show()
