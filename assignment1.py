from assignment1.knn import KNN
from assignment1.dmc import DMC
from helpers.csv_helper import train_test_split
from helpers.dataset import Dataset
from helpers.realization import Realization
from helpers.scores import Scores
from helpers.math import mean, standard_deviation

# Import plotting modules, if they're available
try:
    from helpers.plot_helper import plot_decision_surface
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False

# Requirements
# Minimum: Python 3.9.2
# Optional (only required for plotting graphs):
#   numpy 1.20.2
#   matplotlib 3.3.4


def evaluate(model, dataset, ratio=0.8, num_realizations=20):
    realizations = list()

    for i in range(0, num_realizations):
        classes = list()
        predictions = list()

        # Train the model
        training_set, test_set = train_test_split(dataset.load(), ratio, shuffle=True)
        model.train(training_set)

        # Test the model
        for row in test_set:
            predictions.append(model.predict(row[:-1]))
            classes.append(row[-1])

        scores = Scores(classes, predictions)
        realization = Realization(scores=scores)
        realizations.append(realization)

    accuracies = [r.scores.accuracy for r in realizations]
    m = mean(accuracies)
    std = standard_deviation(accuracies)
    print("Accuracy: {:.2f}% ± {:.2f}%".format(m * 100, std * 100))


def plot_evaluate(model, dataset, ratio=0.8, cols=(0, 1)):
    data = dataset.load()

    # We can only plot 2D
    for i in range(len(data)):
        data[i] = [data[i][cols[0]], data[i][cols[1]], data[i][-1]]

    training_set, test_set = train_test_split(data, ratio, shuffle=True)
    model.train(training_set)

    if type(model) == DMC:
        extra_set = model.training_set  # Centroids
    else:
        extra_set = list()

    features = ["Tamanho da sépala (cm)", "Largura da sépala (cm)", "Tamanho da pétala (cm)", "Largura da pétala (cm)"]
    # features = ["X", "Y"]

    # legend = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    # legend = {0: 'Setosa', 1: 'Outras'}
    # legend = {0: 'Versicolor', 1: 'Outras'}
    # legend = {0: 'Virginica', 1: 'Outras'}
    legend = None
    # legend = {0: '0', 1: '1'}
    plot_decision_surface(model, test_set, extra_set=extra_set, offset=0.2, legend=legend,
                          title="Base de dados Artificial", xlabel=features[cols[0]], ylabel=features[cols[1]])
                          # filename="knn_{}_{}.jpg".format(cols[0], cols[1]))
    print('Done!')


# Iris dataset
iris_encodings = [
    {'Iris-setosa': 0},      # Binary: 0 - Setosa, 1 - Others
    {'Iris-versicolor': 0},  # Binary: 0 - Virginica, 1 - Others
    {'Iris-virginica': 0},   # Binary: 0 - Versicolor, 1 - Others
    {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Multiclass
]
dataset = Dataset('assignment1/datasets/iris.csv', encoding=iris_encodings[1])

# Artificial dataset
# dataset = Dataset('assignment1/datasets/artificial.csv')

# Params
train_test_ratio = 0.8
evaluation_rounds = 20

# KNN
model = KNN(7)

# DMC
# model = DMC()

# Calculate average accuracy for the model
evaluate(model, dataset, ratio=train_test_ratio, num_realizations=evaluation_rounds)

# Plot decision surface
if plotting_available:
    plot_evaluate(model, dataset, ratio=train_test_ratio, cols=(0, 1))
