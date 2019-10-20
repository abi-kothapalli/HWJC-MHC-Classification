from sys import path
import os.path

path.append(os.path.dirname(path[0]))
__package__ = "trainer"

from sklearn.model_selection import GridSearchCV
from models.models import *
from utils.display import display_progress


class Trainer:

    def __init__(self, config, data_loader, evaluator):
        self.config = config
        self.runs = config.runs
        self.model = config.model
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.completed = 0

    def train_one_iteration(self, seed):
        parameters, model = self.get_model(self.model, seed)

        grid = GridSearchCV(model, parameters, cv=self.config.folds, n_jobs=self.config.n_jobs)

        X_train_std, X_test_std = self.data_loader.getX()
        y_train, y_test = self.data_loader.getY()

        grid.fit(X_train_std, y_train)
        best_model = grid.best_estimator_

        self.evaluator.evaluate(best_model, y_test, X_test_std)

    def train(self, seed=100):

        if self.config.active_features:
            parameters, model = self.get_model("l1", seed)
            grid = GridSearchCV(model, parameters, cv=self.config.folds)
            grid.fit(self.data_loader.getX(), self.data_loader.getY())
            coefficients = grid.best_estimator_.coef_[0]

            self.evaluator.evaluate_features(self.data_loader, coefficients)
        else:

            display_progress(0, self.runs)

            for i in range(self.runs-self.completed):

                seed = i + self.completed

                self.data_loader.split_data(seed)
                self.train_one_iteration(seed)

                if self.config.save_checkpoints and (seed+1) % self.config.checkpoint_save_frequency == 0:
                    self.evaluator.checkpoint()

                display_progress(seed + 1, self.runs)

    def set_completed(self, completed):
        self.completed = completed

    def get_model(self, model, seed=None):
        if model == "knn":
            return KNN()
        elif model == "l1":
            return LR(model, seed)
        elif model == "l2":
            return LR(model, seed)
        elif model == "mlp":
            return MLP(seed)
        elif model == "rf":
            return RF(seed)
        elif model == "svm":
            return SVM(seed)
        elif model == "vc":
            return VC(seed)
        else:
            raise NameError(f"Model {model} not found")