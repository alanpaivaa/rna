import math
from assignment4.helpers import step


class SigmoidLogisticActivationFunction:
    @staticmethod
    def transform_class(value):
        if int(value) == 1:
            return 0.99
        return value

    @staticmethod
    def derivative(value):
        return value * (1 - value)

    @staticmethod
    def activate(u):
        return 1 / (1 + math.exp(-1 * u))

    @staticmethod
    def step(u):
        return step(u, threshold=0.5)
