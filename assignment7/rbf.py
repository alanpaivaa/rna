from assignment7.k_means import k_means
from assignment7.helpers import euclidean_distance
import math


class RBF:
    def __init__(self, num_hidden=3, sigma=0.1):
        self.num_hidden = num_hidden
        self.sigma = sigma
        self.h = None

    def predict(self, row):
        return 0

    def phi(self, row, xi):
        dist = euclidean_distance(row, xi)
        return math.exp(-(dist ** 2) / (2 * self.sigma ** 2))

    def generate_hidden(self, training_set):
        self.h = list()
        xi_t = k_means(training_set, self.num_hidden)

        for row in training_set:
            h_row = list()
            for r in xi_t:
                h_row.append(self.phi(row[:-1], r))
            h_row.append(-1)  # Add bias
            self.h.append(h_row)

        print(self.h)

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_hidden(training_set)


