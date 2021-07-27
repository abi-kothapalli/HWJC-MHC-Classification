from os import walk
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import roc_curve
from random import shuffle


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--colors", nargs="*", help="List of matplotlib colors to plot ROC curves")
    parser.add_argument("-pl", "--positive_label", default="MHC", help="Positive label in true classes file")
    args = parser.parse_args()

    if args.colors is None:
        args.colors = get_colors()
    else:
        args.colors.extend(get_colors())

    return args


def plot(args):

    files = []

    for (dirpath, dirnames, filenames) in walk("output/"):
        files.extend(filenames)
        break

    predictions = []
    true_classes = []
    models = []

    for file in files:
        
        name = file.split("_")

        if name[len(name)-1] == "predictions.csv":
            predictions.append(file)
            models.append(name[1].upper())
        else:
            true_classes.append(file)

    predictions.sort()
    true_classes.sort()
    models.sort()

    i = 0
    for prediction, true_class, model in zip(predictions, true_classes, models):
        plot_roc(prediction, true_class, model, args.colors[i], args.positive_label)
        i += 1

    setup()


def plot_roc(predictions, true_classes, label, color, pos):

    input_predictions = pd.read_csv(f"output/{predictions}", header=None)
    input_true_classes = pd.read_csv(f"output/{true_classes}", header=None)

    y_score = np.asarray(input_predictions)
    y_score = [i[0] for i in y_score]
    y_test = np.asarray(input_true_classes)
    y_test = [i[0] for i in y_test]

    fpr, tpr, _ = roc_curve(y_test,  y_score, pos_label=pos)
    plt.plot(fpr, tpr, label=label, color=color)


def setup():
    title = "Receiver Operating Characteristic (ROC) Curve for Voting Classifier and Constituent Models"

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guessing")
    plt.title(title)
    plt.xlim([-0.01, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()


def get_colors():

    color_list = []
    color_dict = dict(colors.BASE_COLORS, **colors.CSS4_COLORS)
    for name, color in color_dict.items():
        color_list.append(name)

    shuffle(color_list)
    return color_list


if __name__ == "__main__":

    config = get_args()
    plot(config)