from knn import KNN
from dmc import DMC
from helpers.csv_helper import load_dataset, train_test_split
from helpers.plot_helper import plot_decision_surface

# TODO: Requirements


def encoded_classes(dataset, encoding):
    # For values not contained in the encoding dict, we use the value max + 1
    unknown = max(encoding.values()) + 1
    for row in dataset:
        klass = row[-1]  # Last element of the array is the class
        if encoding.get(klass) is not None:
            row[-1] = encoding[row[-1]]
        else:
            row[-1] = unknown
    return dataset


def evaluate(model, dataset, ratio=0.8, rounds=1):
    total_accuracy = 0

    for i in range(0, rounds):
        correct_predictions = 0

        # Train the model
        training_set, test_set = train_test_split(dataset, ratio, shuffle=True)
        model.train(training_set)

        # Test the model
        for row in test_set:
            klass = row[-1]
            predicted = model.predict(row)
            if predicted == klass:
                correct_predictions += 1
        total_accuracy += correct_predictions / len(test_set)

    average_accuracy = total_accuracy / rounds
    print("Accuracy: {:.2f}".format(average_accuracy))


def plot_evaluate(model, dataset, ratio=0.8):
    training_set, test_set = train_test_split(dataset, ratio, shuffle=True)
    model.train(training_set)

    if type(model) == DMC:
        extra_set = model.training_set  # Centroids
    else:
        extra_set = list()

    plot_decision_surface(model, test_set, extra_set=extra_set, offset=0.2)
    print('Done!')


# Load dataset
filename = 'iris.csv'
dataset = load_dataset(filename)

# Encode class value from string to integer
encoding = {'Iris-virginica': 0, 'Iris-setosa': 1, 'Iris-versicolor': 2}
dataset = encoded_classes(dataset, encoding)

# Params
train_test_ratio = 0.8
evaluation_rounds = 10

# KNN
# model = KNN(5)

# DMC
model = DMC()

# Calculate average accuracy for the model
evaluate(model, dataset, ratio=train_test_ratio, rounds=evaluation_rounds)

# Plot decision surface
plot_evaluate(model, dataset, ratio=train_test_ratio)
