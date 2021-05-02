from helpers.dataset import generate_regression_dataset, Dataset
from helpers.csv_helper import write_dataset, train_test_split
from helpers.realization import Realization
from helpers.scores import Scores
from assignment3.adaline import Adaline
from helpers.normalizer import normalize
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


def evaluate(model, dataset, ratio=0.8, num_realizations=20, draw_decision_surface=False):
    # max_scores = None
    errors = None
    realizations = list()

    for i in range(0, num_realizations):
        full_dataset = dataset.load()
        # TODO: Should we normalize?
        normalize(full_dataset, include_last_column=True)

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

        # realization.scores = Scores(y, predictions)

        # if max_scores is None or realization.scores.accuracy > max_scores.accuracy:
        #     max_scores = realization.scores
        errors = model.errors

        realization.training_set = training_set
        realization.test_set = test_set

        realizations.append(realization)

    # print("Best accuracy: {:.2f}%".format(max_scores.accuracy * 100))

    # accuracies = list(map(lambda r: r.scores.accuracy, realizations))
    # mean_accuracy = math_helper.mean(accuracies)
    # std_accuracy = math_helper.standard_deviation(accuracies)
    # print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

    # cia = 0  # Closest accuracy index
    # for i in range(1, len(accuracies)):
    #     if abs(mean_accuracy - accuracies[i]) < abs(mean_accuracy - accuracies[cia]):
    #         cia = i

    # if plotting_available:
    #     plt.plot(range(1, len(errors) + 1), errors)
    #     plt.xlabel("Épocas")
    #     plt.ylabel("Soma dos erros")
    #     plt.show()e:
    #     plt.plot(range(1, len(errors) + 1), errors)
    #     plt.xlabel("Épocas")
    #     plt.ylabel("Soma dos erros")
    #     plt.show()

    if plotting_available:
        # Set models with the "mean weights"
        realization = realizations[-1]
        # model.weights = realization.weights # TODO: Take proper realization

        # print(realization.training_set)
        # print(realization.test_set)

        # ***** Old Code
        # import numpy as np
        #
        # points = np.array(realization.training_set + realization.test_set)
        # points_x = points[:, 0]
        # points_y = points[:, 1]
        # plt.scatter(points_x, points_y)
        #
        # space_x = np.linspace(0, 1, 100)
        # space_y = np.array([model.predict([row]) for row in space_x])
        # plt.plot(space_x, space_y, color='g')
        # plt.show()

        plot_regression_surface(model, realization.training_set + realization.test_set)

        plt.show()



# Generate datasets artificial 1 and 2
# generate_datasets()

# Artificial 1
# dataset = Dataset('assignment3/datasets/artificial1.csv')

# Artificial 2
dataset = Dataset('assignment3/datasets/artificial2.csv')

draw_decision_surface = False#"artificial" in dataset.filename

learning_rate = 0.01
ratio = 0.8
epochs = 100

model = Adaline(epochs=epochs, learning_rate=learning_rate, early_stopping=True, verbose=True)
evaluate(model, dataset, ratio=ratio, num_realizations=1, draw_decision_surface=draw_decision_surface)
print("Done!")
