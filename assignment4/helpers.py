def vector_sum(a, b):
    return [i+j for (i, j) in zip(a, b)]


def vector_scalar_product(vector, number):
    return [number * v for v in vector]


def matrix_product(a, b):
    acc = 0
    for (i, j) in zip(a, b):
        acc += (i*j)
    return acc
