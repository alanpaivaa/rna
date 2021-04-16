def normalize_col(vector, col):
    x_min = float('inf')
    x_max = float('-inf')
    for i in range(len(vector)):
        x_min = min(x_min, vector[i][col])
        x_max = max(x_max, vector[i][col])
    for i in range(len(vector)):
        vector[i][col] = (vector[i][col] - x_min) / (x_max - x_min)


def normalize(vector, include_last_column=True):
    if include_last_column:
        offset = 0
    else:
        offset = 1
    for i in range(len(vector[0]) - offset):
        normalize_col(vector, i)
