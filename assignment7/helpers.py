import random
import math
from csv import reader as csv_reader
from csv import writer as csv_writer
from csv import QUOTE_MINIMAL
from functools import reduce


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


def write_dataset(dataset, filename):
    with open(filename, mode='w') as file:
        dataset_writer = csv_writer(file, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL)
        for row in dataset:
            dataset_writer.writerow(row)


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


def train_test_split(dataset, ratio=0.8, shuffle=False):
    dataset_copy = dataset.copy()
    if shuffle:
        random.shuffle(dataset_copy)
    train_index = int(len(dataset_copy) * ratio)
    training_set = dataset_copy[:train_index]
    test_set = dataset_copy[train_index:]
    return training_set, test_set


def standard_deviation(vector):
    # Avoid zero division
    if len(vector) == 1:
        return 0
    m = mean(vector)
    mean_difference = map(lambda x: (x - m) ** 2, vector)
    return math.sqrt(reduce(lambda x, y: x + y, mean_difference) / (len(vector) - 1))


def shape(tensor):
    l0 = len(tensor)
    l1 = len(tensor[0])
    return l0, l1


def matrix_t(a):
    rows, cols = shape(a)
    result = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = a[i][j]
    return result


def matrix_product(a, b):
    shape_a, shape_b = shape(a), shape(b)
    assert shape_a[1] == shape_b[0]

    result = [[0 for _ in range(shape_b[1])] for _ in range(shape_a[0])]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]

    return result


def step(u, threshold=0):
    if u >= threshold:
        return 1
    else:
        return 0


def generate_one_hot_encodings(dataset):
    max_class = -1
    for row in dataset:
        max_class = max(max_class, row[-1])
    num_classes = max_class + 1

    one_hot_encodings = dict()
    for i in range(num_classes):
        encoding = [0] * num_classes
        encoding[i] = 1
        one_hot_encodings[i] = encoding

    return one_hot_encodings


def one_hot_encode(dataset):
    encodings = generate_one_hot_encodings(dataset)

    result = list()
    for row in dataset:
        klass = row[-1]
        encoding = encodings[klass]
        one_hot = list()
        for i in encoding:
            one_hot.append(i)
        result.append(one_hot)

    return result


def class_from_probs(u_t, y_t):
    count = 0
    for y in y_t:
        count += y

    # In "doubt" area
    if count != 1:
        i_max = 0
        for i in range(len(u_t)):
            if u_t[i] > u_t[i_max]:
                i_max = i
        for i in range(len(u_t)):
            if i == i_max:
                y_t[i] = 1
            else:
                y_t[i] = 0

    for i in range(len(y_t)):
        if y_t[i] == 1:
            return i

    assert False, "Should return one of the classes"
