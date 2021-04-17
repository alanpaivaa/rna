import random
from helpers.math import vector_sum, vector_scalar_product, matrix_product, step


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.errors = None
        self.early_stopping = early_stopping
        self.verbose = verbose

    @staticmethod
    def criterion(d, y):
        return d - y

    @staticmethod
    def biased_row(row):
        # Replace the class column with wo value (for bias term)
        biased_row = row.copy()
        biased_row += [-1]
        return biased_row

    def initialize_weights(self, num_weights):
        self.weights = [random.uniform(0, 1) for _ in range(num_weights)]

    def optimize_weights(self, row, error):
        # n * e * x
        nex = vector_scalar_product(self.biased_row(row), error * self.learning_rate)
        # w = w + (n * e * x)
        self.weights = vector_sum(self.weights, nex)

    def predict(self, row):
        # Calculate activation function and output
        u = matrix_product(self.biased_row(row), self.weights)
        return step(u)

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def train(self, training_set):
        # Initialize weights
        # Last column is considered to be the class value
        num_features = len(training_set[0]) - 1
        num_weights = num_features + 1
        self.initialize_weights(num_weights)
        self.errors = list()

        for epoch in range(self.epochs):
            error_sum = 0
            random.shuffle(training_set)

            for row in training_set:
                # Make prediction
                y = self.predict(row[:-1])

                # Calculate error
                error = self.criterion(row[-1], y)
                error_sum += abs(error)

                # Update weights with the learning rule
                self.optimize_weights(row[:-1], error)

            self.log("epoch {}, error: {:.2f}".format(epoch, error_sum))
            self.errors.append(error_sum)

            # No errors in the epoch, no need to continue training
            if self.early_stopping and error_sum == 0:
                self.log("Early stopping training in epoch {} as no mistakes were made".format(epoch))
                break
