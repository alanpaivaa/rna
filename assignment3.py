import random
from assignment3.helpers import sample_points
from assignment3.helpers import write_dataset
from assignment3.dataset import Dataset
from assignment3.perceptron_network import PerceptronNetwork

# Import plotting modules, if they're available
try:
    from helpers.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def generate_artificial_dataset():
    num_samples = 50
    space_size = 1000
    circles = sample_points(num_samples=num_samples, x_range=(0, 5), y_range=(5, 10), space_size=space_size)
    stars = sample_points(num_samples=num_samples, x_range=(5, 10), y_range=(0, 5), space_size=space_size)
    triangles = sample_points(num_samples=num_samples, x_range=(10, 15), y_range=(5, 10), space_size=space_size)

    # Plot points
    if plotting_available:
        circles_x = [point[0] for point in circles]
        circles_y = [point[1] for point in circles]
        plt.scatter(circles_x, circles_y, marker="o")

        stars_x = [point[0] for point in stars]
        stars_y = [point[1] for point in stars]
        plt.scatter(stars_x, stars_y, marker="*")

        triangles_x = [point[0] for point in triangles]
        triangles_y = [point[1] for point in triangles]
        plt.scatter(triangles_x, triangles_y, marker="^")

        plt.show()

    dataset = [point + [0] for point in circles] + \
              [point + [1] for point in stars] + \
              [point + [2] for point in triangles]

    random.shuffle(dataset)

    write_dataset(dataset, "assignment3/datasets/artificial.csv")


# Generate artificial dataset
# generate_artificial_dataset()

dataset = Dataset("assignment3/datasets/artificial.csv")

model = PerceptronNetwork()
model.train(dataset.load())

print("Done!")
