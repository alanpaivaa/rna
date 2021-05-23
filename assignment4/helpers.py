import math
import random
from functools import reduce
from csv import reader as csv_reader


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv_reader(file)
        for row in reader:
            dataset.append(row)
    return dataset


def load_dataset(filename):
    dataset = load_csv(filename)
    for row in dataset:
        for i in range(len(row)):
            # Best effort: we just skip in case we can't convert a row to number
            try:
                row[i] = float(row[i])
            except ValueError:
                pass
    return dataset


def vector_sum(a, b):
    return [i+j for (i, j) in zip(a, b)]


def vector_scalar_product(vector, number):
    return [number * v for v in vector]


def matrix_product(a, b):
    acc = 0
    for (i, j) in zip(a, b):
        acc += (i*j)
    return acc


def step(u, threshold=0):
    if u >= threshold:
        return 1
    else:
        return 0


def train_test_split(dataset, ratio=0.8, shuffle=False):
    dataset_copy = dataset.copy()
    if shuffle:
        random.shuffle(dataset_copy)
    train_index = int(len(dataset_copy) * ratio)
    training_set = dataset_copy[:train_index]
    test_set = dataset_copy[train_index:]
    return training_set, test_set


def mean(vector):
    return reduce(lambda x, y: x + y, vector) / len(vector)


def standard_deviation(vector):
    # Avoid zero division
    if len(vector) == 1:
        return 0
    m = mean(vector)
    mean_difference = map(lambda x: (x - m) ** 2, vector)
    return math.sqrt(reduce(lambda x, y: x + y, mean_difference) / (len(vector) - 1))
