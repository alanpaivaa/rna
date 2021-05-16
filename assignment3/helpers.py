from csv import writer as csv_writer
from csv import reader as csv_reader
from csv import QUOTE_MINIMAL
import random


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv_reader(file)
        for row in reader:
            dataset.append(row)
    return dataset


def load_dataset(filename):
    dataset = load_csv(filename)
    for row in dataset:
        for i in range(len(row)):
            # Best effort: we just skip in case we can't convert a row to number
            try:
                row[i] = float(row[i])
            except ValueError:
                pass
    return dataset

def write_dataset(dataset, filename):
    with open(filename, mode='w') as file:
        dataset_writer = csv_writer(file, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL)
        for row in dataset:
            dataset_writer.writerow(row)


def linear_space(start, end, size):
    space = (end - start) / (size - 1)
    return [start] + [start + space * i for i in range(1, size)]


def sample_points(num_samples, x_range, y_range, space_size):
    # Generate x and y possible values
    x_values = linear_space(x_range[0], x_range[1], space_size)
    y_values = linear_space(y_range[0], y_range[1], space_size)

    # Make coordinates as combination of x and y values
    coordinates = [[round(x, 2), round(y, 2)] for x in x_values for y in y_values]

    # Sample coordinates
    return random.choices(coordinates, k=num_samples)