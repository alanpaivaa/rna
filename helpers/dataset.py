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


def generate_regression_dataset(num_samples, ranges, coefficients, noise=.0):
    # Generate linear spaces
    lin_spaces = list()
    space_size = 1000
    for r in ranges:
        space = (r[1] - r[0]) / (space_size - 1)
        lin_spaces.append([r[0]] + [r[0] + space * i for i in range(1, space_size)])

    # Round decimal places of lin space
    # lin_spaces = [[round(number, 2) for number in row] for row in lin_spaces]

    # Generates coordinates based on each lin space axes
    coordinates = list()
    generate_coordinates(lin_spaces, 0, [], coordinates)

    # Sample coordinates from lin space
    random.shuffle(coordinates)
    coordinates = coordinates[:num_samples]

    # Generate last coordinate based on function coefficients
    y = list()
    for coordinate in coordinates:
        fx = matrix_product(coefficients, coordinate + [1])
        fx += random.uniform(-noise, noise)
        # fx = round(fx, 2)
        y.append(fx)

    dataset = [row_x + [row_y] for row_x, row_y in zip(coordinates, y)]
    return dataset


def generate_coordinates(lin_spaces, i, curr, res):
    if i >= len(lin_spaces):
        res.append(curr.copy())
        return

    for x in lin_spaces[i]:
        curr.append(x)
        generate_coordinates(lin_spaces, i + 1, curr, res)
        curr.pop()


