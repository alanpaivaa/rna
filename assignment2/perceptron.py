import random
import helpers.math as math_helper


class Perceptron:
    def __init__(self, learning_rate=00.1, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

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
        # random.uniform(0, 1)
        self.weights = [0 for _ in range(num_weights)]

    def optimize_weights(self, row, loss):
        self.weights = math_helper.vector_sum(
            self.weights,
            math_helper.vector_scalar_product(self.biased_row(row), loss * self.learning_rate)
        )

    def predict(self, row):
        # Calculate activation function and output
        u = math_helper.matrix_product(self.biased_row(row), self.weights)
        return math_helper.step(u)

    def train(self, training_set):
        # Initialize weights
        # Last column is considered to be the class value
        num_features = len(training_set[0]) - 1
        num_weights = num_features + 1
        self.initialize_weights(num_weights)

        for epoch in range(self.epochs):
            loss_sum = 0
            for row in training_set:
                # Make prediction
                y = self.predict(row[:-1])

                # Calculate loss
                loss = self.criterion(row[-1], y)
                loss_sum += abs(loss)

                # Update weights with the learning rule
                self.optimize_weights(row[:-1], loss)

            print("Epoch {}:\nLoss: {:.2f}\n".format(epoch, loss_sum))

            # No errors in the epoch, no need to continue training
            if loss_sum == 0:
                print("Early stopping training in epoch {} as no mistakes were made".format(epoch))
                break
