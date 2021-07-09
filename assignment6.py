import random
from assignment6.helpers import train_test_split, mean, standard_deviation
from assignment6.dataset import Dataset
from assignment6.multi_layer_perceptron import MultiLayerPerceptron
from assignment6.realization import Realization
from assignment6.scores import Scores, RegressionScores
from assignment6.normalizer import Normalizer
from assignment6.activation_functions import LinearActivationFunction, LogisticActivationFunction
from assignment6.activation_functions import HyperbolicTangentActivationFunction

# Import plotting modules, if they're available
try:
    from assignment6.plot_helper import plot_decision_surface, plot_regression_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


# def select_hyper_parameters(dataset, activation_function, k=5):
#     random.shuffle(dataset)
#     fold_size = int(len(dataset) / k)
#
#     epochs = [25, 50, 100, 200, 300, 400, 500]
#     learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
#     results = list()
#
#     for epoch in epochs:
#         for learning_rate in learning_rates:
#             realizations = list()
#             for i in range(k):
#                 test_start = i * fold_size
#                 test_end = (i + 1) * fold_size
#
#                 # Make training and test sets
#                 training_set = list()
#                 test_set = list()
#                 for j in range(len(dataset)):
#                     if j < test_start or j >= test_end:
#                         training_set.append(dataset[j].copy())
#                     else:
#                         test_set.append(dataset[j].copy())
#
#                 model = GeneralPerceptronNetwork(activation_function, learning_rate=learning_rate, epochs=epoch)
#                 model.train(training_set)
#
#                 d = list()
#                 y = list()
#
#                 # Validate the model
#                 for row in test_set:
#                     d.append(row[-1])
#                     y.append(model.predict(row[:-1]))
#
#                 realization = Realization(training_set, test_set, None, Scores(d, y), None)
#                 realizations.append(realization)
#
#             accuracies = list(map(lambda r: r.scores.accuracy, realizations))
#             mean_accuracy = mean(accuracies)
#             print("Epochs: {}     Learning rate: {}     Accuracy: {:.2f}%".format(epoch, learning_rate, mean_accuracy * 100))
#
#             results.append((epoch, learning_rate, mean_accuracy))
#
#     results = sorted(results, key=lambda r: r[2], reverse=True)
#     best_hyper_parameters = results[0]
#     print("\n\n>>> Best hyper parameters:")
#     print("Epochs: {}     Learning rate: {}     Accuracy: {:.2f}%".format(best_hyper_parameters[0], best_hyper_parameters[1], best_hyper_parameters[2] * 100))


def evaluate(model, dataset, regression=False, ratio=0.8, num_realizations=20):
    normalizer = Normalizer()
    normalizer.fit(dataset)

    if regression:
        normalized_dataset = [normalizer.normalize(row) for row in dataset]
    else:
        normalized_dataset = [normalizer.normalize(row[:-1]) + [row[-1]] for row in dataset]

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
        if regression:
            realization = Realization(training_set,
                                      test_set,
                                      model.layers,
                                      RegressionScores(y, d),
                                      model.errors)
        else:
            realization = Realization(training_set,
                                      test_set,
                                      model.layers,
                                      Scores(d, y),
                                      model.errors)
        realizations.append(realization)

    if regression:
        # Sort realizations by mse
        realizations = sorted(realizations, key=lambda r: r.scores.mse)

        # MSE Stats
        mses = list(map(lambda r: r.scores.mse, realizations))
        avg_mse = mean(mses)
        std_mse = standard_deviation(mses)
        print("MSE: {:.5f} ± {:.5f}".format(avg_mse, std_mse))

        # RMSE Stats
        rmses = list(map(lambda r: r.scores.rmse, realizations))
        avg_rmse = mean(rmses)
        std_rmse = standard_deviation(rmses)
        print("RMSE: {:.5f} ± {:.5f}".format(avg_rmse, std_rmse))

        # Realization whose mse is closest to the mean
        avg_realization = sorted(realizations, key=lambda r: abs(avg_mse - r.scores.mse))[0]

        # Plot error sum plot
        if plotting_available:
            plt.plot(range(1, len(avg_realization.errors) + 1), avg_realization.errors)
            plt.xlabel("Épocas")
            plt.ylabel("Soma dos erros")
            plt.show()

        # Plot decision surface
        if plotting_available:
            # Set models with the "mean weights"
            model.layers = avg_realization.layers
            plot_regression_surface(model,
                                    normalizer,
                                    avg_realization.training_set + avg_realization.test_set,
                                    x_label="X", y_label="Y",
                                    scatter_label='Base de dados',
                                    model_label='Predição do modelo', title='Artificial I')
    else:
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
            model.layers = avg_realization.layers
            plot_decision_surface(model,
                                  normalized_dataset,
                                  title="Superfície de Decisão",
                                  xlabel="X1",
                                  ylabel="X2")


# Dataset descriptors (lazy loaded)
# Artificial
artificial_dataset = Dataset("assignment6/datasets/artificial.csv")

# Iris
iris_dataset = Dataset("assignment6/datasets/iris.csv")

# Vertebral column
column_dataset = Dataset("assignment6/datasets/vertebral-column.csv")

# Dermatology
dermatology_dataset = Dataset("assignment6/datasets/dermatology.csv")

# Breast Cancer
breast_cancer_dataset = Dataset("assignment6/datasets/breast-cancer.csv")

# Sin
artificial_regression_dataset = Dataset("assignment6/datasets/artificial-regression.csv", regression=True)

# Best hyper parameter found using grid search with k-fold cross validation
hyper_parameters = {
    'artificial': (artificial_dataset, False, 500, 0.3),
    'iris': (iris_dataset, False, 600, 0.1),
    'column': (column_dataset, False, 600, 0.05),
    'dermatology': (dermatology_dataset, False, 600, 0.1),
    'breast_cancer': (breast_cancer_dataset, False, 600, 0.1),
    'artificial_regression': (artificial_regression_dataset, True, 1000, 0.1),
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

dataset, regression, epochs, learning_rate = hyper_parameters['artificial']

split_ratio = 0.8
num_realizations = 1

model = MultiLayerPerceptron(num_hidden=7,
                             regression=regression,
                             learning_rate=learning_rate,
                             epochs=epochs,
                             early_stopping=True,
                             verbose=False)
evaluate(model, dataset.load(), regression=regression, ratio=split_ratio, num_realizations=num_realizations)

print("Done!")
