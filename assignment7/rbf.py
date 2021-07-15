from assignment7.k_means import k_means
from assignment7.helpers import euclidean_distance, matrix_t, matrix_product, one_hot_encode, class_from_probs
from assignment7.activation_functions import LogisticActivationFunction
import math
import numpy as np


class RBF:
    def __init__(self, num_hidden=None, regression=False):
        self.num_hidden = num_hidden
        self.h = None
        self.xi_t = None
        self.weights = None
        self.sigmas = None
        self.regression = regression

    @staticmethod
    def rbf(row, xi, sigma):
        dist = euclidean_distance(row, xi)
        return math.exp(-((dist ** 2) / (2 * sigma ** 2)))

    def generate_hidden(self, training_set):
        self.h = list()
        self.xi_t = k_means(training_set, self.num_hidden)

        d_max = np.max([euclidean_distance(c1, c2) for c1 in self.xi_t for c2 in self.xi_t])
        self.sigmas = [d_max / math.sqrt(2 * self.num_hidden)] * self.num_hidden

        for row in training_set:
            h_row = list()
            for r in range(len(self.xi_t)):
                h_row.append(self.rbf(row[:-1], self.xi_t[r], self.sigmas[r]))
            h_row.append(-1)
            self.h.append(h_row)

    def generate_olam_weights(self, training_set):
        if self.regression:
            d = [[row[-1]] for row in training_set]
        else:
            d = one_hot_encode(training_set)
        ht = matrix_t(self.h)
        ht_h = matrix_product(ht, self.h)
        ht_h_inv = np.linalg.inv(np.array(ht_h))
        ht_h_inv = ht_h_inv.tolist()
        ht_h_inv_ht = matrix_product(ht_h_inv, ht)
        self.weights = matrix_product(ht_h_inv_ht, d)

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.h = None
        self.xi_t = None
        self.weights = None
        self.sigmas = None

        self.generate_hidden(training_set)
        self.generate_olam_weights(training_set)

    def predict(self, row):
        h_row = list()
        for r in range(len(self.xi_t)):
            h_row.append(self.rbf(row, self.xi_t[r], self.sigmas[r]))
        h_row.append(-1)  # Bias

        u_t = matrix_product([h_row], self.weights)[0]

        if self.regression:
            return u_t[0]

        # Classification
        f = LogisticActivationFunction()
        probs = [f.activate(u) for u in u_t]
        y_t = [f.step(prob) for prob in probs]

        return class_from_probs(u_t, y_t)
