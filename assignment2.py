import matplotlib.pyplot as plt
from helpers.dataset import Dataset
from assignment2.perceptron import Perceptron
from helpers.csv_helper import train_test_split
import helpers.math as math_helper
from helpers.scores import Scores


def evaluate(model, dataset, ratio=0.8, realizations=20):
    accuracies = list()

    max_accuracy = 0
    losses = None

    for i in range(0, realizations):
        # Train the model
        training_set, test_set = train_test_split(dataset.load(), ratio, shuffle=True)
        model.train(training_set)

        y = list()
        predictions = list()

        # Test the model
        for row in test_set:
            y.append(row[-1])
            predictions.append(model.predict(row[:-1]))

        scores = Scores(y, predictions)
        accuracies.append(scores.accuracy)

        scores.print_confusion_matrix()

        if scores.accuracy > max_accuracy:
            max_accuracy = scores.accuracy
            losses = model.losses

    mean_accuracy = math_helper.mean(accuracies)
    std_accuracy = math_helper.standard_deviation(accuracies)
    print("Mean accuracy: {:.2f}%".format(mean_accuracy * 100))
    print("Accuracy standard deviation: {:.2f}%".format(std_accuracy * 100))

    plt.plot(losses)
    plt.title("Soma dos erros por época pra realização com maior taxa de acerto")
    plt.xlabel("Épocas")
    plt.ylabel("Soma dos erros")
    plt.show()


# Iris dataset
iris_encodings = [
    {'Iris-setosa': 0},      # Binary: 0 - Setosa, 1 - Others
    {'Iris-versicolor': 0},  # Binary: 0 - Virginica, 1 - Others
    {'Iris-virginica': 0},   # Binary: 0 - Versicolor, 1 - Others
    {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Multiclass (doesn't make sense for perceptron)
]
dataset = Dataset('assignment2/datasets/iris.csv', encoding=iris_encodings[1])

ratio = 0.8

model = Perceptron(epochs=50, early_stopping=False)
evaluate(model, dataset, ratio=ratio, realizations=20)

