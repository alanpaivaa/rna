import math
from assignment4.helpers import step


class LogisticActivationFunction:
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


class HyperbolicTangentActivationFunction:
    @staticmethod
    def transform_class(d):
        if int(d) == 1:
            return 0.99
        return -0.99

    @staticmethod
    def derivative(y):
        return 0.5 * (1 - y ** 2)

    @staticmethod
    def activate(u):
        return (1 - math.exp(-u)) / (1 + math.exp(-u))

    @staticmethod
    def step(u):
        return step(u, threshold=0)
