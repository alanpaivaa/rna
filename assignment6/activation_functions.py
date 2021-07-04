import math
from assignment6.helpers import step


class LinearActivationFunction:
    @staticmethod
    def transform_d(d):
        return d

    @staticmethod
    def transform_y(y):
        return step(y)

    @staticmethod
    def derivative(value):
        return 1

    @staticmethod
    def activate(u):
        return u

    @staticmethod
    def step(u):
        return step(u)

    @staticmethod
    def step_train():
        return True


class LogisticActivationFunction:
    @staticmethod
    def transform_d(d):
        if int(d) == 1:
            return 0.99
        return 0.01

    @staticmethod
    def transform_y(y):
        return y

    @staticmethod
    def derivative(value):
        return value * (1 - value)

    @staticmethod
    def activate(u):
        return 1 / (1 + math.exp(-1 * u))

    @staticmethod
    def step(u):
        return step(u, threshold=0.5)

    @staticmethod
    def step_train():
        return False


class HyperbolicTangentActivationFunction:
    @staticmethod
    def transform_d(d):
        if int(d) == 1:
            return 0.99
        return -0.99

    @staticmethod
    def transform_y(y):
        return y

    @staticmethod
    def derivative(y):
        return 0.5 * (1 - y ** 2)

    @staticmethod
    def activate(u):
        return (1 - math.exp(-u)) / (1 + math.exp(-u))

    @staticmethod
    def step(u):
        return step(u)

    @staticmethod
    def step_train():
        return False
