import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, config, split=True, pretrained=False):
        self.config = config
        self.split = split
        self.pretrained = pretrained

        self.df = pd.read_csv(f"data/{config.data_file}")

        if not pretrained:
            X = self.df.drop(labels=config.target_feature, axis=1)
            self.X_array = np.asarray(X)
            self.features = list(X)

            y = self.df[config.target_feature]
            self.y_array = np.asarray(y)

            if not split:
                sc = StandardScaler()
                self.X_array_std = sc.fit_transform(self.X_array)

    def setup(self, features, target_feature):

        if target_feature in self.df.columns:
            X = self.df.drop(labels=target_feature, axis=1)
            self.X_array = np.asarray(X)
            self.features = list(X)
            self.features.sort()
            features.sort()

            if self.features == features:
                y = self.df[target_feature]
                self.y_array = np.asarray(y)

                sc = StandardScaler()
                self.X_array_std = sc.fit_transform(self.X_array)

            else:
                raise ValueError("Model was trained on different features than input features")

        else:
            self.features = list(self.df)
            self.features.sort()
            features.sort()

            if self.features == features:
                self.X_array = np.asarray(self.df)
                sc = StandardScaler()
                self.X_array_std = sc.fit_transform(self.X_array)
            else:
                raise ValueError("Model was trained on different features than input features")

    def split_data(self, seed):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X_array, self.y_array, test_size=self.config.test_size, random_state=seed)
        sc = StandardScaler()
        self.X_train_std = sc.fit_transform(X_train)
        self.X_test_std = sc.transform(X_test)

    def getX(self):
        if self.split:
            return self.X_train_std, self.X_test_std
        else:
            return self.X_array_std

    def getY(self):
        if self.split:
            return self.y_train, self.y_test
        else:
            return self.y_array

    def getDF(self):
        return self.df

    def getFeatures(self):
        return self.features
