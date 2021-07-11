import matplotlib.pyplot as plt
import numpy as np


def plot_decision_surface(model, dataset, title=None, xlabel=None, ylabel=None, legend=None):
    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    grid = np.array([[x, y] for x in x_values for y in y_values])
    predictions = np.array([model.predict(list(row)) for row in grid])

    np_dataset = np.array(dataset)

    classes = np.unique(predictions)
    colors = ["cornflowerblue", "forestgreen", "salmon"]
    markers = ["o", "*", "^"]
    for c in classes:
        points = grid[predictions == c]
        plt.scatter(points[:, 0], points[:, 1], marker="s", color=colors[c], alpha=0.05)
        set_points = np_dataset[np_dataset[:, -1] == c]
        plt.scatter(set_points[:, 0], set_points[:, 1], color=colors[c], marker=markers[c])

    plt.margins(x=0, y=0)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if legend is not None:
        plt.legend()

    plt.show()


def plot_regression_surface(model, normalizer, dataset, x_label='X', y_label='Y', z_label='Z', scatter_label=None, model_label=None, title=None):
    dataset = np.array(dataset)
    dataset = dataset[dataset[:, 0].argsort()]

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
        rows = [[x[i], y[i], z[i]] for i in range(len(x))]
        ax.scatter3D(rows[:, 0], rows[:, 1], rows[:, 2], label=scatter_label)
    else:
        rows = np.array([[a, b] for a, b in zip(x, y)])
        ax.plot(rows[:, 0], rows[:, 1], label=scatter_label)

    if is_3d:
        space_x, space_y, space_z = list(), list(), list()
        for sx in np.linspace(np.min(x), np.max(x), 100):
            for sy in np.linspace(np.min(y), np.max(y), 100):
                space_x.append(sx)
                space_y.append(sy)
                space_z.append(model.predict([sx, sy]))
        rows = [[space_x[i], space_y[i], space_z[i]] for i in range(len(space_x))]
        rows = np.array([normalizer.denormalize(row) for row in rows])
        ax.plot3D(rows[:, 0], rows[:, 1], rows[:, 2], color=(0, .5, 0, .3), label=model_label)
    else:
        space_x = np.linspace(np.min(x), np.max(x), 1000)
        space_y = np.array([model.predict([row]) for row in space_x])

        rows = np.array([[a, b] for a, b in zip(space_x, space_y)])
        ax.plot(rows[:, 0], rows[:, 1], color='green', label=model_label)

    plt.title(title)
    plt.legend()
    plt.show()
