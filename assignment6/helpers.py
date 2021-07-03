def shape(tensor):
    l0 = len(tensor)
    l1 = len(tensor[0])
    return l0, l1


def step(u, threshold=0):
    if u >= threshold:
        return 1
    else:
        return 0


def matrix_product(a, b):
    shape_a, shape_b = shape(a), shape(b)
    assert shape_a[1] == shape_b[0]

    result = [[0 for _ in range(shape_b[1])] for _ in range(shape_a[0])]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]

    return result
