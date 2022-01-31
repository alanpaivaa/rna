import random
from assignment6.helpers import train_test_split, mean, standard_deviation
from assignment6.dataset import Dataset
from assignment6.multi_layer_perceptron import MultiLayerPerceptron
from assignment6.mlp import MLPTorch
from assignment6.realization import Realization
from assignment6.scores import Scores, RegressionScores
from assignment6.normalizer import Normalizer

# Import plotting modules, if they're available
try:
    from assignment6.plot_helper import plot_decision_surface, plot_regression_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def select_hyper_parameters(dataset, k=5):
    random.shuffle(dataset)
    fold_size = int(len(dataset) / k)

    hidden_layers = range(3, 10)
    epochs = [300, 400, 500, 750, 1000, 1250]
    learning_rate = 0.1
    results = list()

    for num_hidden in hidden_layers:
        for epoch in epochs:
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

                model = MultiLayerPerceptron(
                    num_hidden=num_hidden,
                    regression=False,
                    learning_rate=learning_rate,
                    epochs=epoch,
                    early_stopping=True,
                    verbose=False
                )
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
            print(
                "Hidden: {}     Epochs: {}     Learning rate: {}     Accuracy: {:.2f}%".format(
                    num_hidden, epoch, learning_rate, mean_accuracy * 100
                )
            )

            results.append((epoch, learning_rate, mean_accuracy))

    results = sorted(results, key=lambda r: r[2], reverse=True)
    best_hyper_parameters = results[0]
    print("\n\n>>> Best hyper parameters:")
    print("Epochs: {}     Learning rate: {}     Accuracy: {:.2f}%".format(best_hyper_parameters[0], best_hyper_parameters[1], best_hyper_parameters[2] * 100))


def evaluate(model, dataset, regression=False, normalize=True, ratio=0.8, num_realizations=20):
    if normalize:
        normalizer = Normalizer()
        normalizer.fit(dataset)

        if regression:
            normalized_dataset = [normalizer.normalize(row) for row in dataset]
        else:
            normalized_dataset = [normalizer.normalize(row[:-1]) + [row[-1]] for row in dataset]
    else:
        normalized_dataset = dataset

    realizations = list()
    for i in range(0, num_realizations):
        print("Realization {}...".format(i + 1))
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
                                      model.model.parameters(),
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
        if len(dataset[0][:-1]) == 2 and plotting_available:
            # Set models with the "mean weights"
            model.layers = avg_realization.layers
            plot_regression_surface(model,
                                    normalizer,
                                    avg_realization.training_set + avg_realization.test_set,
                                    x_label="X", y_label="Y",
                                    scatter_label='Base de dados',
                                    model_label='Predição do modelo',
                                    title="{} épocas".format(epochs))
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
            model.model.parameters = avg_realization.layers
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

# Artificial
artificial_regression_dataset = Dataset("assignment6/datasets/artificial-regression.csv", regression=True)

# Abalone
abalone_dataset = Dataset("assignment6/datasets/abalone.csv", regression=True)

# Car
car_dataset = Dataset("assignment6/datasets/car.csv", regression=True)

# Motor
motor_dataset = Dataset("assignment6/datasets/motor.csv", regression=True)

# Best hyper parameter found using grid search with k-fold cross validation
hyper_parameters = {
    'artificial': (artificial_dataset, False, 500, 0.3, 7),
    'iris': (iris_dataset, False, 600, 0.1, 7),
    'column': (column_dataset, False, 600, 0.05, 7),
    'dermatology': (dermatology_dataset, False, 600, 0.1, 7),
    'breast_cancer': (breast_cancer_dataset, False, 600, 0.1, 7),
    'artificial_regression': (artificial_regression_dataset, True, 500, 0.1, 7),
    'abalone': (abalone_dataset, True, 500, 0.1, 7),
    'car': (car_dataset, True, 350, 0.1, 7),
    'motor': (motor_dataset, True, 10, 0.1, 4),
}

# Select best hyper parameters
# datasets = ['artificial', 'iris', 'column', 'dermatology', 'breast_cancer']
# for ds in datasets:
#     print(">>>>>>>>>>>>>> {}".format(ds))
#     dataset, _, _, _ = hyper_parameters['artificial']
#     select_hyper_parameters(dataset.load())
#     print("\n\n\n\n\n")

dataset, regression, epochs, learning_rate, hidden_layers = hyper_parameters['breast_cancer']

split_ratio = 0.8
num_realizations = 20

model = MLPTorch(num_hidden=hidden_layers,
                 batch_size=5,
                 regression=regression,
                 learning_rate=learning_rate,
                 epochs=epochs,
                 verbose=False)
evaluate(model,
         dataset.load(),
         regression=regression,
         normalize=not regression,
         ratio=split_ratio,
         num_realizations=num_realizations)

print("Done!")
