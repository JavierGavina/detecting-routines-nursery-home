import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import Union
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import yaml
from sklearn.preprocessing import LabelBinarizer

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def setup() -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def get_groundtruth_locations(groundtruth: pd.DataFrame) -> dict:
    locations_detected_per_user = dict()
    for user in groundtruth["User"].unique():
        locations_detected_per_user[user] = groundtruth[groundtruth["User"] == user]["Location"].unique().tolist()
    return locations_detected_per_user


def extend_estimated_table(estimated: pd.DataFrame, groundtruth: pd.DataFrame):
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
    groundtruth = pd.read_csv(path, sep=";").iloc[:-1, :]
    groundtruth["WeekDay"] = pd.Categorical(groundtruth["WeekDay"],
                                            categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                                        "Saturday", "Sunday"], ordered=True)
    return groundtruth.sort_values(["User", "Location", "WeekDay", "Start", "End"])


def get_estimated_from_path(path_to_estimated: str, path_to_groundtruth: str) -> pd.DataFrame:
    groundtruth = get_groundtruth_from_path(path_to_groundtruth)
    estimated = pd.read_csv(path_to_estimated, sep=",")
    extended = extend_estimated_table(estimated, groundtruth)
    extended["WeekDay"] = pd.Categorical(extended["WeekDay"],
                                         categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                                                     "Sunday"], ordered=True)
    return extended.sort_values(["User", "Location", "WeekDay", "Start", "End"])


def classificate(data: pd.DataFrame, labelize: bool = False, threshold=0.5) -> pd.DataFrame:
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


def get_probabilities(data: pd.DataFrame) -> pd.DataFrame:
    unique_intervals = data[['Start', 'End']].drop_duplicates()
    user = data["User"].values[0]
    result = pd.DataFrame(columns=["User", "WeekDay", "Start", "End", "Location", "Probability"])
    for weekday in WEEKDAYS:
        for start, end in zip(unique_intervals["Start"], unique_intervals["End"]):
            query = data.query(f"Start=='{start}' and End=='{end}' and WeekDay=='{weekday}'")
            estimated = query.iloc[np.argmax(query.RelativeFrequency)]
            result.loc[len(result)] = [user, weekday, start, end, estimated["Location"], estimated["RelativeFrequency"]]

    return result


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def analyze_results(path_to_estimated: str, path_to_groundtruth: str, user: str = "02A8", threshold=0.5):
    estimated = get_estimated_from_path(path_to_estimated, path_to_groundtruth)
    groundtruth = get_groundtruth_from_path(path_to_groundtruth)
    estimated_user = estimated.query(f"User=='{user}'")
    groundtruth_user = groundtruth.query(f"User=='{user}'")
    estimated_labeled = classificate(estimated_user, labelize=True, threshold=threshold)
    groundtruth_labeled = classificate(groundtruth_user, labelize=True, threshold=threshold)

    common_labels = np.union1d(groundtruth_labeled["Location"].unique(), estimated_labeled["Location"].unique())

    print("CLASSIFICATION REPORT")
    print(classification_report(y_true=groundtruth_labeled["Location"],
                                y_pred=estimated_labeled["Location"],
                                zero_division=0))

    plot_confusion_matrix(confusion_matrix(groundtruth_labeled["Location"], estimated_labeled["Location"]),
                          common_labels)


def labelize_and_probabilities(data: pd.DataFrame, groundtruth: pd.DataFrame) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
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


def get_roc_auc(labels_groundtruth, user_probabilities, classes):
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
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


def plot_roc(user, dificulty):
    # EASY
    groundtruth = os.path.join(config["data_dir"], "groundtruth.csv")

    path_estimated = {
        "easy": os.path.join(config["results_dir"], "easy_frequency_table.csv"),
        "medium": os.path.join(config["results_dir"], "medium_frequency_table.csv"),
        "hard": os.path.join(config["results_dir"], "hard_frequency_table.csv")
    }

    data = get_estimated_from_path(path_estimated[dificulty], groundtruth)
    estimated_user = data.query(f"User=='{user}'")
    groundtruth_user = get_groundtruth_from_path(groundtruth).query(f"User=='{user}'")
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

    user_statistics = pd.DataFrame(columns=["User", "Dificulty", "F1-Macro", "F1-Weighted", "F1-micro", "Macro-AUC", "Micro-AUC"])
    rooms_statistics = pd.DataFrame(columns=["Dificulty", "User", "Room", "F1-Score", "Precision", "Recall", "AUC"])

    for dificulty in ["easy", "medium", "hard"]:

        groundtruth = get_groundtruth_from_path(PATH_GROUNDTRUTH)
        dificulty_data = get_estimated_from_path(eval(f"PATH_{dificulty.upper()}"), PATH_GROUNDTRUTH)
        plt.subplots(2, 3, figsize=(20, 10))

        for user in config["users"]:
            estimated_user = dificulty_data.query(f"User=='{user}'")
            groundtruth_user = groundtruth.query(f"User=='{user}'")
            classes = np.union1d(estimated_user["Location"].unique(), groundtruth_user["Location"].unique())
            labels_groundtruth, user_labels, user_probabilities = labelize_and_probabilities(estimated_user,
                                                                                             groundtruth_user)
            fpr, tpr, roc_auc = get_roc_auc(labels_groundtruth, user_probabilities, classes)
            if len(user_labels) > len(classes):
                classes = np.append(classes, "missclassified")

            for idx, c in enumerate(classes[:-1]):
                f1_room = np.round(f1_score(labels_groundtruth[:, idx], user_labels[:, idx], zero_division=0), 3)
                precision_room = np.round(
                    precision_score(labels_groundtruth[:, idx], user_labels[:, idx], zero_division=0), 3)
                recall_room = np.round(recall_score(labels_groundtruth[:, idx], user_labels[:, idx], zero_division=0),
                                       3)
                auc_room = np.round(roc_auc[idx], 3)
                rooms_statistics.loc[len(rooms_statistics)] = [dificulty, user, c, f1_room, precision_room, recall_room,
                                                               auc_room]

            f1_user = np.round(f1_score(y_true=labels_groundtruth, y_pred=user_labels, average="macro", zero_division=0), decimals=3)
            f1_weighted = np.round(f1_score(y_true=labels_groundtruth, y_pred=user_labels, average="weighted", zero_division=0), decimals=3)
            f1_micro = np.round(f1_score(y_true=labels_groundtruth, y_pred=user_labels, average="micro", zero_division=0), decimals=3)
            macro_auc = np.round(roc_auc["macro"], decimals=3)
            micro_auc = np.round(roc_auc["micro"], decimals=3)
            user_statistics.loc[len(user_statistics)] = [user, dificulty, f1_user, f1_weighted, f1_micro, macro_auc,
                                                         micro_auc]

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

    for dificulty in config["dificulties"]:
        groundtruth = get_groundtruth_from_path(PATH_GROUNDTRUTH)
        dificulty_data = get_estimated_from_path(eval(f"PATH_{dificulty.upper()}"), PATH_GROUNDTRUTH)
        plt.subplots(2, 3, figsize=(20, 10))
        for user in config["users"]:
            estimated_user = dificulty_data.query(f"User=='{user}'")
            groundtruth_user = groundtruth.query(f"User=='{user}'")
            classes = np.union1d(estimated_user["Location"].unique(), groundtruth_user["Location"].unique())
            labels_groundtruth, user_labels, user_probabilities = labelize_and_probabilities(estimated_user,
                                                                                             groundtruth_user)
            if len(user_labels) > len(classes):
                classes = np.append(classes, "missclassified")

            cm = confusion_matrix(y_true=classes[labels_groundtruth.argmax(axis=1)],
                                  y_pred=classes[user_labels.argmax(axis=1)])

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
            # plt.legend('', frameon=False)
        plt.savefig(os.path.join(PATH_CONFUSION_MATRIX, f"confusion_matrix_{dificulty}.png"))
        plt.show()
