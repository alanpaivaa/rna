import matplotlib.pyplot as plt
from helpers.dataset import Dataset
from assignment2.perceptron import Perceptron
from helpers.csv_helper import train_test_split
import helpers.math as math_helper
from helpers.scores import Scores
from helpers.normalizer import normalize


def evaluate(model, dataset, ratio=0.8, realizations=20):
    accuracies = list()

    max_scores = None
    errors = None

    for i in range(0, realizations):
        full_dataset = dataset.load()
        normalize(full_dataset, include_last_column=False)

        # Train the model
        training_set, test_set = train_test_split(full_dataset, ratio, shuffle=True)
        model.train(training_set)

        y = list()
        predictions = list()
        scores = None

        # Test the model
        for row in test_set:
            y.append(row[-1])
            predictions.append(model.predict(row[:-1]))

        scores = Scores(y, predictions)
        accuracies.append(scores.accuracy)

        if max_scores is None or scores.accuracy > max_scores.accuracy:
            max_scores = scores
            errors = model.errors

    print("Best confusion matrix")
    max_scores.print_confusion_matrix()

    # print("Best accuracy: {:.2f}%".format(max_scores.accuracy * 100))

    mean_accuracy = math_helper.mean(accuracies)
    std_accuracy = math_helper.standard_deviation(accuracies)
    print("Mean accuracy: {:.2f}%".format(mean_accuracy * 100))
    print("Accuracy standard deviation: {:.2f}%".format(std_accuracy * 100))

    plt.plot(errors)
    plt.title("Soma dos erros por época")
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

# sonar_encodings = [
#     {'R': 0, 'M': 1}
# ]
# dataset = Dataset('assignment2/datasets/sonar.csv', encoding=sonar_encodings[0])

ratio = 0.8

model = Perceptron(epochs=500, early_stopping=False, verbose=False)
evaluate(model, dataset, ratio=ratio, realizations=20)

