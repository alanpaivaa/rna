from csv import reader as csv_reader
from functools import reduce
import math


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv_reader(file)
        for row in reader:
            dataset.append(row)
    return dataset


def load_dataset(filename, last_column):
    dataset = load_csv(filename)
    offset = 1
    if last_column:
        offset = 0
    for row in dataset:
        for i in range(len(row) - offset):
            # Best effort: we just skip in case we can't convert a row to number
            try:
                row[i] = float(row[i])
            except ValueError:
                row[i] = 0
    return dataset


def euclidean_distance(x, y):
    summation = 0
    count = min(len(x), len(y))
    for i in range(count):
        summation += (x[i] - y[i]) ** 2
    return math.sqrt(summation)


def mean(vector):
    return reduce(lambda x, y: x + y, vector) / len(vector)


def vectors_equal(x, y, tolerance=0):
    return euclidean_distance(x, y) <= tolerance
