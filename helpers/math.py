from functools import reduce
import math


def matrix_product(a, b):
    acc = 0
    for (i, j) in zip(a, b):
        acc += (i*j)
    return acc


def step(u):
    if u >= 0:
        return 1
    else:
        return 0


def vector_scalar_product(vector, number):
    return [number * v for v in vector]


def vector_sum(a, b):
    return [i+j for (i, j) in zip(a, b)]


def mean(vector):
    return reduce(lambda x, y: x + y, vector) / len(vector)


def standard_deviation(vector):
    # Avoid zero division
    if len(vector) == 1:
        return 0
    m = mean(vector)
    mean_difference = map(lambda x: (x - m) ** 2, vector)
    return math.sqrt(reduce(lambda x, y: x + y, mean_difference) / (len(vector) - 1))
