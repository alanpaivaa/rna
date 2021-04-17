import matplotlib.pyplot as plt
from helpers.dataset import Dataset
from assignment2.perceptron import Perceptron
from helpers.csv_helper import train_test_split
import helpers.math as math_helper
from helpers.scores import Scores
from helpers.normalizer import normalize
from helpers.plot_helper import plot_decision_surface
from helpers.realization import Realization


def evaluate(model, dataset, ratio=0.8, num_realizations=20, draw_decision_surface=False):
    max_scores = None
    errors = None
    realizations = list()

    for i in range(0, num_realizations):
        full_dataset = dataset.load()
        normalize(full_dataset, include_last_column=False)

        # Train the model
        training_set, test_set = train_test_split(full_dataset, ratio, shuffle=True)
        model.train(training_set)

        realization = Realization()
        realization.weights = model.weights

        y = list()
        predictions = list()

        # Test the model
        for row in test_set:
            y.append(row[-1])
            predictions.append(model.predict(row[:-1]))

        realization.scores = Scores(y, predictions)

        if max_scores is None or realization.scores.accuracy > max_scores.accuracy:
            max_scores = realization.scores
            errors = model.errors

        if draw_decision_surface:
            realization.training_set = training_set
            realization.test_set = test_set

        realizations.append(realization)

    print("Best confusion matrix")
    max_scores.print_confusion_matrix()

    # print("Best accuracy: {:.2f}%".format(max_scores.accuracy * 100))

    accuracies = list(map(lambda r: r.scores.accuracy, realizations))
    mean_accuracy = math_helper.mean(accuracies)
    std_accuracy = math_helper.standard_deviation(accuracies)
    print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

    if not draw_decision_surface:
        plt.plot(errors)
        plt.title("Soma dos erros por época")
        plt.xlabel("Épocas")
        plt.ylabel("Soma dos erros")
        plt.show()

    if draw_decision_surface:
        cia = 0  # Closest accuracy index
        for i in range(1, len(accuracies)):
            if abs(mean_accuracy - accuracies[i]) < abs(mean_accuracy - accuracies[cia]):
                cia = i

        # Set models with the "mean weights"
        model.weights = realizations[cia].weights

        plot_decision_surface(model, realizations[cia].training_set + realizations[cia].test_set, offset=0.2)


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

# dataset = Dataset('assignment2/datasets/artificial.csv')

draw_decision_surface = "artificial" in dataset.filename

ratio = 0.8

model = Perceptron(epochs=500, learning_rate=0.01, early_stopping=False, verbose=False)
evaluate(model, dataset, ratio=ratio, num_realizations=20, draw_decision_surface=draw_decision_surface)

