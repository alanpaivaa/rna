from csv import reader, writer, QUOTE_MINIMAL
import random


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
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
        dataset_writer = writer(file, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL)
        for row in dataset:
            dataset_writer.writerow(row)


def train_test_split(dataset, ratio=0.8, shuffle=False):
    dataset_copy = dataset.copy()
    if shuffle:
        random.shuffle(dataset_copy)
    train_index = int(len(dataset_copy) * ratio)
    training_set = dataset_copy[:train_index]
    test_set = dataset_copy[train_index:]
    return training_set, test_set
