from helpers.dataset import generate_regression_dataset
from helpers.csv_helper import write_dataset


def generate_datasets():
    # Artificial 1
    artificial1 = generate_regression_dataset(num_samples=40, coefficients=[2.71, 1.82], x_noise=.25, y_noise=15)
    write_dataset(artificial1, 'assignment3/datasets/artificial1.csv')

    # Artificial 2
    artificial2 = generate_regression_dataset(num_samples=40, coefficients=[3.92, 2.44, 1.67], x_noise=.25, y_noise=15)
    write_dataset(artificial2, 'assignment3/datasets/artificial2.csv')


# Generate datasets artificial 1 and 2
generate_datasets()


