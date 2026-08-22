from .._backend import predict


class LinearRegression:

    def __init__(self, weight=1.0, bias=0.0):
        self.weight = weight
        self.bias = bias

    def predict(self, X):
        return [
            predict(float(x), self.weight, self.bias)
            for x in X
        ]