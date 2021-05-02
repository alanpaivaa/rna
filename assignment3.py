from helpers.dataset import generate_regression_dataset, Dataset
from helpers.csv_helper import write_dataset, train_test_split
from helpers.realization import Realization
from helpers.scores import RegressionScores
from assignment3.adaline import Adaline
from helpers.normalizer import Normalizer
from helpers.math import mean, standard_deviation
from helpers.plot_helper import plot_regression_surface

# Import plotting modules, if they're available
try:
    from helpers.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def generate_datasets():
    # Artificial 1
    artificial1 = generate_regression_dataset(num_samples=50,
                                              ranges=[(50, 100)],
                                              coefficients=[2.71, 1.82],
                                              noise=10)
    write_dataset(artificial1, 'assignment3/datasets/artificial1.csv')

    # Artificial 2
    artificial2 = generate_regression_dataset(num_samples=40,
                                              ranges=[(10, 50), (100, 150)],
                                              coefficients=[3.92, 2.44, 1.67],
                                              noise=25)
    write_dataset(artificial2, 'assignment3/datasets/artificial2.csv')


def evaluate(model, dataset, ratio=0.8, num_realizations=20):
    # Setup normalizer
    normalizer = Normalizer()
    normalizer.fit(dataset)
    normalized_dataset = [normalizer.normalize(row) for row in dataset]

    realizations = list()

    for i in range(0, num_realizations):
        # Train the model
        training_set, test_set = train_test_split(normalized_dataset, ratio, shuffle=True)
        model.train(training_set)

        y = list()
        predictions = list()

        # Test the model
        for row in test_set:
            y.append(row[-1])
            predictions.append(model.predict(row[:-1]))

        # Caching realization values
        realization = Realization(training_set,
                                  test_set,
                                  model.weights,
                                  RegressionScores(y, predictions),
                                  model.errors)
        realizations.append(realization)

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
        model.weights = avg_realization.weights
        plot_regression_surface(model, normalizer, avg_realization.training_set + avg_realization.test_set)


# Generate datasets artificial 1 and 2
# generate_datasets()

# Artificial 1
dataset = Dataset('assignment3/datasets/artificial1.csv')

# Artificial 2
# dataset = Dataset('assignment3/datasets/artificial2.csv')

learning_rate = 0.01
ratio = 0.8
epochs = 75

model = Adaline(epochs=epochs, learning_rate=learning_rate, early_stopping=True, verbose=False)
evaluate(model, dataset.load(), ratio=ratio, num_realizations=20)
print("Done!")
