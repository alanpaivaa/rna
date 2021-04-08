import matplotlib.pyplot as plt
import numpy as np


def plot_decision_surface(model, test_set, extra_set=list(), offset=0.0, title=None, xlabel=None, ylabel=None, legend=None, filename=None):
    # Set figure size
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(600 * px, 360 * px))

    # Get x and y out of test set
    np_test_set = np.array(test_set)
    x = np_test_set[:, :-1]
    y = np_test_set[:, -1]

    # Get the X boundaries
    x1_min = np.min(x[:, 0]) - offset
    x1_max = np.max(x[:, 0]) + offset

    # Get the Y boundaries
    x2_min = np.min(x[:, 1]) - offset
    x2_max = np.max(x[:, 1]) + offset

    # Get the meshgrid
    grid_x1, grid_x2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.020),
        np.arange(x2_min, x2_max, 0.020)
    )

    # Get predictions for all points in the mesh grid
    points = np.array([grid_x1.ravel(), grid_x2.ravel()]).T
    predicted = list()
    for row in points:
        predicted.append(model.predict(row))
    predicted = np.array(predicted)

    # Get class information
    classes = np.sort(np.unique(predicted).astype('int'))
    vmin = classes[0]
    vmax = classes[-1]

    colors = ['blue', 'green', 'red', 'purple', 'yellow', 'brown', 'gray']

    # Draw the filled contour
    predicted = np.reshape(predicted, grid_x1.shape)
    plt.contourf(
        grid_x1,
        grid_x2,
        predicted,
        vmax - vmin,
        vmin=vmin,
        vmax=vmax,
        alpha=0.25,
        colors=colors
    )

    # Plot scattered points
    for i in classes:
        idx = y == i
        xi = x[idx]
        if legend is not None:
            label = legend[i]
        else:
            label = None
        plt.scatter(xi[:, 0], xi[:, 1], c=colors[i], s=10, label=label)

    # Plot extra points
    np_extra_set = np.array(extra_set)
    if len(extra_set) > 0:
        for i in classes:
            idx = np_extra_set[:, -1] == i
            xi = np_extra_set[idx]
            if legend is not None:
                label = legend[i]
            else:
                label = None
            plt.scatter(xi[:, 0], xi[:, 1], c=colors[i], s=100, marker=',', label=label)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    # Draw legend
    if legend is not None:
        plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
