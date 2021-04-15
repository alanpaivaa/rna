from helpers.dataset import Dataset
from assignment2.perceptron import Perceptron
from helpers.csv_helper import train_test_split
import helpers.math as math_helper


def evaluate(model, dataset, ratio=0.8, realizations=20):
    accuracies = list()

    for i in range(0, realizations):
        correct_predictions = 0

        # Train the model
        training_set, test_set = train_test_split(dataset.load(), ratio, shuffle=True)
        model.train(training_set)

        # Test the model
        for row in test_set:
            predicted = model.predict(row[:-1])
            if predicted == row[-1]:
                correct_predictions += 1
        accuracies.append(correct_predictions / len(test_set))

    mean_accuracy = math_helper.mean(accuracies)
    std_accuracy = math_helper.standard_deviation(accuracies)
    print("Mean accuracy: {:.2f}%".format(mean_accuracy * 100))
    print("Accuracy standard deviation: {:.2f}%".format(std_accuracy * 100))

# Iris dataset
iris_encodings = [
    {'Iris-setosa': 0},      # Binary: 0 - Setosa, 1 - Others
    {'Iris-versicolor': 0},  # Binary: 0 - Virginica, 1 - Others
    {'Iris-virginica': 0},   # Binary: 0 - Versicolor, 1 - Others
    {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Multiclass (doesn't make sense for perceptron)
]
dataset = Dataset('assignment2/datasets/iris.csv', encoding=iris_encodings[2])

ratio = 0.8

model = Perceptron(epochs=50)
evaluate(model, dataset, ratio=ratio)

