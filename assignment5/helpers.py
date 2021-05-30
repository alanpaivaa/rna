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


def matrix_elementwise_product(a, b):
    result = list()
    for i in range(len(a)):
        result.append(a[i].copy())
        for j in range(len(a[i])):
            result[i][j] = a[i][j] * b[i][j]
    return result


def matrix_t(a):
    rows, cols = shape(a)
    result = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = a[i][j]
    return result
