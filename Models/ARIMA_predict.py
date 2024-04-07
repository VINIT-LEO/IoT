import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMA_predictor(object):
    def __init__(self, order=(1, 0, 0)):
        self.predictor = ARIMA(order=order)

    def fit_predictor(self, train_np):
        num_samples = np.shape(train_np)[0]
        train = np.empty((0,), dtype='float32')
        for i in range(num_samples):
            data = train_np[i][5]
            train = np.append(train, data)
        self.predictor.fit(train)

    def predict_outlier(self, data):
        sample = data[-1][5]
        prediction = self.predictor.forecast(steps=1)
        return int(sample > prediction)
