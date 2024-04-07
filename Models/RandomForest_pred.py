import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest_predictor(object):
    def __init__(self):
        self.predictor = RandomForestClassifier(n_estimators=100)

    def fit_predictor(self, train_np):
        num_samples = np.shape(train_np)[0]
        train = np.empty((0, 4), dtype='float32')
        for i in range(1, num_samples):
            data = np.array([(train_np[i-1][0]-train_np[i][0]), (train_np[i-1][3]-train_np[i][3]),
                            (train_np[i-1][4]-train_np[i][4]), train_np[i][5]]).reshape(1, -1)
            train = np.append(train, data, axis=0)
        labels = np.zeros(num_samples-1, dtype=int)
        labels[np.where(train_np[1:, 5] > train_np[:-1, 5])] = 1
        self.predictor.fit(train, labels)

    def predict_outlier(self, data):
        sample = np.array([(data[-2][0]-data[-1][0]), (data[-1][3]-data[-2][3]),
                          (data[-1][4]-data[-2][4]), data[-1][5]]).reshape(1, -1)
        prediction = self.predictor.predict(sample)
        return prediction[0]
