from assignment2.perceptron import Perceptron
from helpers.math import matrix_product


class Adaline(Perceptron):
    def __init__(self, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        super().__init__(learning_rate, epochs, early_stopping, verbose)

    def predict(self, row):
        return matrix_product(self.biased_row(row), self.weights)
