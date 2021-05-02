import matplotlib.pyplot as plt
import numpy as np


def plot_decision_surface(model, test_set, extra_set=list(), offset=0.0, title=None, xlabel=None, ylabel=None, legend=None, filename=None):
    # Set figure size
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(600 * px, 400 * px))

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
        predicted.append(model.predict(list(row)))
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
            plt.scatter(xi[:, 0], xi[:, 1], c=colors[i], s=100, marker=',')

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


def plot_regression_surface(model, dataset, x_label='X', y_label='Y', z_label='Z'):
    dataset = np.array(dataset)

    is_3d = dataset.shape[1] == 3

    # Get x and y columns
    x = dataset[:, 0]
    y = dataset[:, 1]

    z = None
    if is_3d:
        z = dataset[:, 2]

    if is_3d:
        ax = plt.axes(projection="3d")
    else:
        ax = plt.axes()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if is_3d:
        ax.set_zlabel(z_label)

    # Draw dataset points
    if is_3d:
        ax.scatter3D(x, y, z)
    else:
        ax.scatter(x, y)

    if is_3d:
        space_x, space_y, space_z = list(), list(), list()
        for sx in np.linspace(np.min(x), np.max(x), 100):
            for sy in np.linspace(np.min(y), np.max(y), 100):
                space_x.append(sx)
                space_y.append(sy)
                space_z.append(model.predict([sx, sy]))
        ax.plot3D(space_x, space_y, space_z, color=(0, .5, 0, .3))
    else:
        space_x = np.linspace(np.min(x), np.max(x), 1000)
        space_y = np.array([model.predict([row]) for row in space_x])
        ax.plot(space_x, space_y, color='green')

    plt.show()
