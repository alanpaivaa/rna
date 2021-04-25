from helpers.csv_helper import load_dataset
from helpers.math import vector_sum, matrix_product
import random


class Dataset:
    def __init__(self, filename, encoding=None):
        self.filename = filename
        self.encoding = encoding

    def encoded_classes(self, dataset):
        # For values not contained in the encoding dict, we use the value max + 1
        unknown = max(self.encoding.values()) + 1
        for row in dataset:
            klass = row[-1]  # Last element of the array is the class
            if self.encoding.get(klass) is not None:
                row[-1] = self.encoding[row[-1]]
            else:
                row[-1] = unknown
        return dataset

    @staticmethod
    def set_last_column_int(dataset):
        for row in dataset:
            row[-1] = int(row[-1])

    def load(self):
        dataset = load_dataset(self.filename)
        if self.encoding is not None:
            dataset = self.encoded_classes(dataset)
        self.set_last_column_int(dataset)
        return dataset


def generate_regression_dataset(num_samples, coefficients, x_step=1, x_noise=.5, y_noise=5):
    dims = len(coefficients) - 1

    x = [[x_step + x_noise for _ in range(dims)]]
    while len(x) < num_samples:
        last_x = x[-1]
        steps = [x_step + random.uniform(0, x_noise) for _ in range(dims)]
        x.append(vector_sum(last_x, steps))
    x = [[round(number, 2) for number in row] for row in x]

    y = list()
    for row in x:
        fx = matrix_product(coefficients, row + [1])
        fx += random.uniform(-y_noise, y_noise)
        fx = round(fx, 2)
        y.append(fx)

    dataset = [row_x + [row_y] for row_x, row_y in zip(x, y)]
    return dataset
