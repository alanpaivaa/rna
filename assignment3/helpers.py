from csv import writer as csv_writer
from csv import reader as csv_reader
from csv import QUOTE_MINIMAL
import random
from functools import reduce
import math


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


def write_dataset(dataset, filename):
    with open(filename, mode='w') as file:
        dataset_writer = csv_writer(file, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL)
        for row in dataset:
            dataset_writer.writerow(row)


def linear_space(start, end, size):
    space = (end - start) / (size - 1)
    return [start] + [start + space * i for i in range(1, size)]


def sample_points(num_samples, x_range, y_range, space_size):
    # Generate x and y possible values
    x_values = linear_space(x_range[0], x_range[1], space_size)
    y_values = linear_space(y_range[0], y_range[1], space_size)

    # Make coordinates as combination of x and y values
    coordinates = [[round(x, 2), round(y, 2)] for x in x_values for y in y_values]

    # Sample coordinates
    return random.choices(coordinates, k=num_samples)


def shape(tensor):
    l0 = len(tensor)
    l1 = len(tensor[0])
    return l0, l1


def matrix_product(a, b):
    shape_a, shape_b = shape(a), shape(b)
    assert shape_a[1] == shape_b[0]

    result = [[0 for _ in range(shape_b[1])] for _ in range(shape_a[0])]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]

    return result


def step(u):
    if u >= 0:
        return 1
    else:
        return 0


def matrix_scalar_product(scalar, matrix):
    return [[scalar * number for number in row] for row in matrix]


def matrix_add(a, b):
    result = [[0 for _ in row] for row in a]
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = a[i][j] + b[i][j]
    return result


def matrix_sub(a, b):
    return matrix_add(a, matrix_scalar_product(-1, b))


def matrix_t(a):
    rows, cols = shape(a)
    result = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = a[i][j]
    return result


def mean(vector):
    return reduce(lambda x, y: x + y, vector) / len(vector)


def standard_deviation(vector):
    # Avoid zero division
    if len(vector) == 1:
        return 0
    m = mean(vector)
    mean_difference = map(lambda x: (x - m) ** 2, vector)
    return math.sqrt(reduce(lambda x, y: x + y, mean_difference) / (len(vector) - 1))


def train_test_split(dataset, ratio=0.8, shuffle=False):
    dataset_copy = dataset.copy()
    if shuffle:
        random.shuffle(dataset_copy)
    train_index = int(len(dataset_copy) * ratio)
    training_set = dataset_copy[:train_index]
    test_set = dataset_copy[train_index:]
    return training_set, test_set
