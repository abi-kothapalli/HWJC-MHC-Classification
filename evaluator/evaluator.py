from sys import path
import os.path

path.append(os.path.dirname(path[0]))
__package__ = "evaluator"

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from statistics import median
from utils.save_load import save_checkpoint, save_model, save_output

np.set_printoptions(formatter={"float_kind": "{:f}".format})


class Evaluator:

    def __init__(self, config):
        self.config = config
        self.auc_scores = []
        self.accuracies = []
        self.prob_predictions = []
        self.true_classes = []

        if self.config.save_model:
            self.models = []
        else:
            self.models = None

    def evaluate(self, model, y, X):

        yhat = model.predict(X)
        yhat_probabilities = model.predict_proba(X)
        yhat_probabilities = [i[1] for i in yhat_probabilities]

        self.auc_scores.append(roc_auc_score(y, yhat_probabilities))
        self.accuracies.append(accuracy_score(y, yhat))
        self.true_classes.append(y.tolist())
        self.prob_predictions.append(yhat_probabilities)

        if self.config.save_model:
            self.models.append(model)

    def evaluate_pretrianed(self, model, data_loader, target_feature):

        yhat = model.predict(data_loader.getX())
        yhat_probabilities = model.predict_proba(data_loader.getX())
        yhat_probabilities = [i[1] for i in yhat_probabilities]

        if target_feature in data_loader.getDF().columns:
            auc = roc_auc_score(data_loader.getY(), yhat_probabilities)
            acc = accuracy_score(data_loader.getY(), yhat)

            print(f"AUC: {auc}")
            print(f"Accuracy: {acc}")

        else:
            data_frame = data_loader.getDF()
            data_frame = data_frame.assign(target_feature=yhat, Probability=yhat_probabilities)

            file = f"output/pretrained_predictions/{self.config.experiment_name}_with_{self.config.model}.csv"
            data_frame.to_csv(f"{file}", index=False)
            print(f"Predictions complete: saved as {file}")

    def evaluate_features(self, data_loader, coefficients):

        numCoef = len(coefficients)
        counts = [0] * numCoef
        for i in range(numCoef):
            if coefficients[i] != 0:
                counts[i] += 1

        features = data_loader.getFeatures()
        df_active_features = data_loader.getDF()
        for i in range(len(features)):
            if counts[i] == 0:
                df_active_features.drop(labels=features[i], axis=1, inplace=True)
            else:
                print(f"{features[i]}: {coefficients[i]}")

        filename = self.config.data_file[:-4]
        df_active_features.to_csv(f"data/{filename}_active_features.csv", index=False)

    def get_auc(self):
        if self.config.evaluation_mode == "max":
            return max(self.auc_scores)
        else:
            return median(self.auc_scores)

    def get_accuracy(self):
        if self.config.evaluation_mode == "max":
            return max(self.accuracies)
        else:
            return median(self.accuracies)

    def checkpoint(self):
        save_checkpoint(self.config, self.auc_scores, self.accuracies, self.prob_predictions, self.true_classes)

    def set_checkpoint(self, auc, acc, pred, classes):
        self.auc_scores = auc
        self.accuracies = acc
        self.prob_predictions = pred
        self.true_classes = classes

    def save(self, features=None):

        save_output(self.config, self.prob_predictions, self.true_classes, self.auc_scores)

        if self.config.save_model:

            if self.config.evaluation_mode == "max":
                target_auc = max(self.auc_scores)
            else:
                if self.config.runs % 2 == 0:
                    new_auc = self.auc_scores.copy()
                    new_auc.sort()
                    target_auc = new_auc[int(self.config.runs / 2)]
                else:
                    target_auc = median(self.auc_scores)

            index = self.auc_scores.index(target_auc)
            save_model(self.config, self.models[index], features)