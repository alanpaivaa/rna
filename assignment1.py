from assignment1.knn import KNN
from assignment1.dmc import DMC
from helpers.csv_helper import train_test_split
from helpers.dataset import Dataset

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
            predicted = model.predict(row[:-1])
            if predicted == klass:
                correct_predictions += 1
        total_accuracy += correct_predictions / len(test_set)

    average_accuracy = total_accuracy / rounds
    print("Accuracy: {:.2f}%".format(average_accuracy * 100))


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
dataset = Dataset('assignment1/datasets/iris.csv', encoding=iris_encodings[3])

# Artificial dataset
# dataset = Dataset('assignment1/datasets/artificial.csv')

# Params
train_test_ratio = 0.8
evaluation_rounds = 10

# KNN
model = KNN(7)

# DMC
# model = DMC()

# Calculate average accuracy for the model
evaluate(model, dataset, ratio=train_test_ratio, rounds=evaluation_rounds)

# Plot decision surface
if plotting_available:
    plot_evaluate(model, dataset, ratio=train_test_ratio, cols=(0, 1))
