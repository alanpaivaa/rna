from helpers.dataset import Dataset
from assignment1.perceptron import Perceptron
from helpers.csv_helper import train_test_split
import helpers.math as math_helper
from helpers.scores import Scores
from helpers.normalizer import normalize
from helpers.realization import Realization

# Import plotting modules, if they're available
try:
    from helpers.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False

# Requirements
# Minimum: Python 3.9.2
# Optional (only required for plotting graphs):
#   numpy 1.20.2
#   matplotlib 3.3.4


def select_training_hyper_parameters(dataset, ratio=0.8, num_folds=5):
    training_set, _ = train_test_split(dataset.load(), ratio, shuffle=True)

    folds = list()
    fold_size = int(len(training_set) / num_folds)
    for i in range(num_folds):
        fold = [training_set[i] for i in range(i*fold_size, (i+1) * fold_size)]
        folds.append(fold)

    learning_rate = 0.01
    epochs = 50

    for i in range(50):
        # learning_rate *= 10
        e = epochs * i

        realizations = list()

        for i in range(num_folds):
            training_set = list()

            # Flattening folds for training
            for j in range(num_folds):
                if j == i:
                    continue
                training_set += folds[j].copy()

            model = Perceptron(epochs=e, learning_rate=learning_rate, early_stopping=False, verbose=False)
            model.train(training_set)

            val_set = folds[i]

            classes = list()
            predictions = list()

            for row in val_set:
                predictions.append(model.predict(row[:-1]))
                classes.append(row[-1])

            scores = Scores(classes, predictions)
            realization = Realization(scores=scores)
            realizations.append(realization)

        accuracies = [r.scores.accuracy for r in realizations]
        mean_accuracy = math_helper.mean(accuracies)
        std_accuracy = math_helper.standard_deviation(accuracies)
        print("Learning rate: {}, Epochs: {}, Accuracy: {:.2f}% ± {:.2f}%".format(learning_rate, e, mean_accuracy * 100, std_accuracy * 100))


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

    # print("Best accuracy: {:.2f}%".format(max_scores.accuracy * 100))

    accuracies = list(map(lambda r: r.scores.accuracy, realizations))
    mean_accuracy = math_helper.mean(accuracies)
    std_accuracy = math_helper.standard_deviation(accuracies)
    print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

    print("Best confusion matrix")
    max_scores.print_confusion_matrix()

    cia = 0  # Closest accuracy index
    for i in range(1, len(accuracies)):
        if abs(mean_accuracy - accuracies[i]) < abs(mean_accuracy - accuracies[cia]):
            cia = i

    print("Confusion matrix closest to the mean accuracy")
    realizations[cia].scores.print_confusion_matrix()

    if plotting_available and not draw_decision_surface:
        plt.plot(range(1, len(errors) + 1), errors)
        plt.xlabel("Épocas")
        plt.ylabel("Soma dos erros")
        plt.show()

    if plotting_available and draw_decision_surface:
        # Set models with the "mean weights"
        model.weights = realizations[cia].weights
        # print([[round(n, 4) for n in row] for row in realizations[cia].training_set])
        # print([[round(n, 4) for n in row] for row in realizations[cia].test_set])

        plot_decision_surface(model, realizations[cia].training_set, realizations[cia].test_set,
                              offset=0.2, xlabel="X", ylabel="Y",
                              title="Perceptron", legend={0: '0', 1: '1'})


# Iris dataset
iris_encodings = [
    {'Iris-setosa': 0},      # Binary: 0 - Setosa, 1 - Others
    {'Iris-versicolor': 0},  # Binary: 0 - Virginica, 1 - Others
    {'Iris-virginica': 0},   # Binary: 0 - Versicolor, 1 - Others
]
dataset = Dataset('assignment1/datasets/iris.csv', encoding=iris_encodings[0])

# dataset = Dataset('assignment1/datasets/artificial.csv')

draw_decision_surface = "artificial" in dataset.filename

learning_rate = 0.01
ratio = 0.8
epochs = 100  # Setosa, virginica, artificial
# epochs = 10   # Versicolor


# select_training_hyper_parameters(dataset)

model = Perceptron(epochs=epochs, learning_rate=learning_rate, early_stopping=True, verbose=False)
evaluate(model, dataset, ratio=ratio, num_realizations=20, draw_decision_surface=draw_decision_surface)

