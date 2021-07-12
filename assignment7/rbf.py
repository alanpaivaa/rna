from assignment7.k_means import k_means
from assignment7.helpers import euclidean_distance, matrix_t, matrix_product
import math
import numpy as np


class RBF:
    def __init__(self, num_hidden=3, sigma=0.1):
        self.num_hidden = num_hidden
        self.sigma = sigma
        self.h = None
        self.weights = None

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

    def generate_weights(self, training_set):
        d = [[row[-1]] for row in training_set]
        ht = matrix_t(self.h)
        ht_h = matrix_product(ht, self.h)
        ht_h_inv = np.linalg.inv(np.array(ht_h))  # TODO: Don't use numpy
        ht_h_inv = ht_h_inv.tolist()
        ht_h_inv_ht = matrix_product(ht_h_inv, ht)
        self.weights = matrix_product(ht_h_inv_ht, d)

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_hidden(training_set)
        self.generate_weights(training_set)
