import random

from assignment5.helpers import train_test_split, mean, standard_deviation
from assignment5.dataset import Dataset
from assignment5.general_perceptron_network import GeneralPerceptronNetwork
from assignment5.realization import Realization
from assignment5.scores import Scores
from assignment5.normalizer import Normalizer
from assignment5.activation_functions import LinearActivationFunction, LogisticActivationFunction, HyperbolicTangentActivationFunction

# Import plotting modules, if they're available
try:
    from assignment5.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def select_hyper_parameters(dataset, activation_function, k=5):
    random.shuffle(dataset)
    fold_size = int(len(dataset) / k)

    epochs = [25, 50, 100, 200, 300, 400, 500]
    learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
    results = list()

    for epoch in epochs:
        for learning_rate in learning_rates:
            realizations = list()
            for i in range(k):
                test_start = i * fold_size
                test_end = (i + 1) * fold_size

                # Make training and test sets
                training_set = list()
                test_set = list()
                for j in range(len(dataset)):
                    if j < test_start or j >= test_end:
                        training_set.append(dataset[j].copy())
                    else:
                        test_set.append(dataset[j].copy())

                model = GeneralPerceptronNetwork(activation_function, learning_rate=learning_rate, epochs=epoch)
                model.train(training_set)

                d = list()
                y = list()

                # Validate the model
                for row in test_set:
                    d.append(row[-1])
                    y.append(model.predict(row[:-1]))

                realization = Realization(training_set, test_set, None, Scores(d, y), None)
                realizations.append(realization)

            accuracies = list(map(lambda r: r.scores.accuracy, realizations))
            mean_accuracy = mean(accuracies)
            print("Epochs: {}     Learning rate: {}     Accuracy: {:.2f}%".format(epoch, learning_rate, mean_accuracy * 100))

            results.append((epoch, learning_rate, mean_accuracy))

    results = sorted(results, key=lambda r: r[2], reverse=True)
    best_hyper_parameters = results[0]
    print("\n\n>>> Best hyper parameters:")
    print("Epochs: {}     Learning rate: {}     Accuracy: {:.2f}%".format(best_hyper_parameters[0], best_hyper_parameters[1], best_hyper_parameters[2] * 100))


def evaluate(model, dataset, ratio=0.8, num_realizations=20):
    normalizer = Normalizer()
    normalizer.fit(dataset)
    normalized_dataset = [normalizer.normalize(row[:-1]) + [row[-1]] for row in dataset]
    # normalized_dataset = dataset

    realizations = list()

    for i in range(0, num_realizations):
        # Train the model
        training_set, test_set = train_test_split(normalized_dataset, ratio, shuffle=True)
        model.train(training_set)

        d = list()
        y = list()

        # Test the model
        for row in test_set:
            d.append(row[-1])
            y.append(model.predict(row[:-1]))

        # Caching realization values
        realization = Realization(training_set,
                                  test_set,
                                  model.weights,
                                  Scores(d, y),
                                  model.errors)
        realizations.append(realization)

    # Accuracy Stats
    accuracies = list(map(lambda r: r.scores.accuracy, realizations))
    mean_accuracy = mean(accuracies)
    std_accuracy = standard_deviation(accuracies)
    print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

    # Realization whose accuracy is closest to the mean
    avg_realization = sorted(realizations, key=lambda r: abs(mean_accuracy - r.scores.accuracy))[0]

    print("Confusion matrix")
    avg_realization.scores.print_confusion_matrix()

    # Plot error sum plot
    if plotting_available:
        plt.plot(range(1, len(avg_realization.errors) + 1), avg_realization.errors)
        # plt.title("Artificial")
        plt.xlabel("Épocas")
        plt.ylabel("Soma dos erros")
        plt.show()

    # Plot decision surface
    if len(dataset[0][:-1]) == 2 and plotting_available:
        # Set models with the "mean weights"
        model.weights = avg_realization.weights
        plot_decision_surface(model,
                              normalized_dataset,
                              title="Superfície de Decisão",
                              xlabel="X1",
                              ylabel="X2")


# Dataset descriptors (lazy loaded)
# Artificial
artificial_dataset = Dataset("assignment5/datasets/artificial.csv")

# Iris
iris_dataset = Dataset("assignment5/datasets/iris.csv")
iris01_dataset = Dataset("assignment5/datasets/iris.csv", features=[0, 1])
iris02_dataset = Dataset("assignment5/datasets/iris.csv", features=[0, 2])
iris03_dataset = Dataset("assignment5/datasets/iris.csv", features=[0, 3])
iris12_dataset = Dataset("assignment5/datasets/iris.csv", features=[1, 2])
iris13_dataset = Dataset("assignment5/datasets/iris.csv", features=[1, 3])
iris23_dataset = Dataset("assignment5/datasets/iris.csv", features=[2, 3])

# Activation functions
linear_activation_function = LinearActivationFunction()
logistic_activation_function = LogisticActivationFunction()
tanh_activation_function = HyperbolicTangentActivationFunction()

# Best hyper parameter found using grid search with k-fold cross validation
hyper_parameters = {
    ('artificial', 'linear'): (artificial_dataset, linear_activation_function, 25, 0.01),
    ('artificial', 'logistic'): (artificial_dataset, logistic_activation_function, 200, 0.05),
    ('artificial', 'tanh'): (artificial_dataset, tanh_activation_function, 200, 0.01),
    ('iris', 'linear'): (iris_dataset, linear_activation_function, 100, 0.1),
    ('iris', 'logistic'): (iris_dataset, logistic_activation_function, 750, 0.01),
    ('iris', 'tanh'): (iris_dataset, tanh_activation_function, 500, 0.01),
    ('iris01', 'linear'): (iris01_dataset, linear_activation_function, 600, 0.001),
    ('iris01', 'logistic'): (iris01_dataset, logistic_activation_function, 400, 0.05),
    ('iris01', 'tanh'): (iris01_dataset, tanh_activation_function, 600, 0.005),
    ('iris02', 'linear'): (iris02_dataset, linear_activation_function, 600, 0.001),
    ('iris02', 'logistic'): (iris02_dataset, logistic_activation_function, 750, 0.05),
    ('iris02', 'tanh'): (iris02_dataset, tanh_activation_function, 300, 0.05),
    ('iris03', 'linear'): (iris03_dataset, linear_activation_function, 1000, 0.005),
    ('iris03', 'logistic'): (iris03_dataset, logistic_activation_function, 400, 0.05),
    ('iris03', 'tanh'): (iris03_dataset, tanh_activation_function, 750, 0.05),
    ('iris12', 'linear'): (iris12_dataset, linear_activation_function, 750, 0.005),
    ('iris12', 'logistic'): (iris12_dataset, logistic_activation_function, 1000, 0.05),
    ('iris12', 'tanh'): (iris12_dataset, tanh_activation_function, 750, 0.01),
    ('iris13', 'linear'): (iris13_dataset, linear_activation_function, 200, 0.01),
    ('iris13', 'logistic'): (iris13_dataset, logistic_activation_function, 400, 0.1),
    ('iris13', 'tanh'): (iris13_dataset, tanh_activation_function, 200, 0.05),
    ('iris23', 'linear'): (iris23_dataset, linear_activation_function, 500, 0.01),
    ('iris23', 'logistic'): (iris23_dataset, logistic_activation_function, 400, 0.05),
    ('iris23', 'tanh'): (iris23_dataset, tanh_activation_function, 500, 0.01),
}

# Select best hyper parameters
# select_hyper_parameters(iris_dataset.load(), tanh_activation_function)
# ds = ['iris01', 'iris02', 'iris03', 'iris12', 'iris13', 'iris23']
# fs = ['linear', 'logistic', 'tanh']
# pairs = [(d, f) for d in ds for f in fs]
# for d, f in pairs:
#     dataset, activation_function, _, _ = hyper_parameters[(d, f)]
#     print(">>>>>>>>>>>>>> {} - {} - {} - {}".format(d, f, dataset.filename, dataset.features))
#     select_hyper_parameters(dataset.load(), activation_function)
#     print("\n\n\n\n\n")

dataset, activation_function, epochs, learning_rate = hyper_parameters[('iris', 'tanh')]

split_ratio = 0.8
num_realizations = 20

model = GeneralPerceptronNetwork(activation_function,
                                 learning_rate=learning_rate,
                                 epochs=epochs,
                                 early_stopping=True,
                                 verbose=False)
evaluate(model, dataset.load(), ratio=split_ratio, num_realizations=num_realizations)

print("Done!")
