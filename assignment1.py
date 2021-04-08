from assignment1.knn import KNN
from assignment1.dmc import DMC
from helpers.csv_helper import train_test_split
from helpers.plot_helper import plot_decision_surface
from assignment1.dataset import Dataset

# TODO: Requirements


def evaluate(model, dataset, ratio=0.8, rounds=1):
    total_accuracy = 0

    for i in range(0, rounds):
        correct_predictions = 0

        # Train the model
        training_set, test_set = train_test_split(dataset.load(), ratio, shuffle=True)
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
    training_set, test_set = train_test_split(dataset.load(), ratio, shuffle=True)
    model.train(training_set)

    if type(model) == DMC:
        extra_set = model.training_set  # Centroids
    else:
        extra_set = list()

    plot_decision_surface(model, test_set, extra_set=extra_set, offset=0.2)
    print('Done!')


# Iris dataset
iris_encodings = [
    {'Iris-setosa': 0},      # Binary: 0 - Setosa, 1 - Others
    {'Iris-versicolor': 0},  # Binary: 0 - Virginica, 1 - Others
    {'Iris-virginica': 0},   # Binary: 0 - Versicolor, 1 - Others
    {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Multiclass
]
dataset = Dataset('assignment1/datasets/iris.csv', encoding=iris_encodings[0])

# Artificial dataset
# dataset = Dataset('assignment1/datasets/artificial.csv')

# Params
train_test_ratio = 0.8
evaluation_rounds = 10

# KNN
model = KNN(5)

# DMC
# model = DMC()

# Calculate average accuracy for the model
evaluate(model, dataset, ratio=train_test_ratio, rounds=evaluation_rounds)

# Plot decision surface
plot_evaluate(model, dataset, ratio=train_test_ratio)
