import random
from assignment3.helpers import sample_points, write_dataset, train_test_split, mean, standard_deviation
from assignment3.dataset import Dataset
from assignment3.perceptron_network import PerceptronNetwork
from assignment3.realization import Realization
from assignment3.scores import Scores
from assignment3.normalizer import Normalizer

# Import plotting modules, if they're available
try:
    from helpers.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def generate_artificial_dataset():
    num_samples = 50
    space_size = 1000
    circles = sample_points(num_samples=num_samples, x_range=(0, 5), y_range=(5, 10), space_size=space_size)
    stars = sample_points(num_samples=num_samples, x_range=(5, 10), y_range=(0, 5), space_size=space_size)
    triangles = sample_points(num_samples=num_samples, x_range=(10, 15), y_range=(5, 10), space_size=space_size)

    # Plot points
    if plotting_available:
        circles_x = [point[0] for point in circles]
        circles_y = [point[1] for point in circles]
        plt.scatter(circles_x, circles_y, marker="o")

        stars_x = [point[0] for point in stars]
        stars_y = [point[1] for point in stars]
        plt.scatter(stars_x, stars_y, marker="*")

        triangles_x = [point[0] for point in triangles]
        triangles_y = [point[1] for point in triangles]
        plt.scatter(triangles_x, triangles_y, marker="^")

        plt.show()

    dataset = [point + [0] for point in circles] + \
              [point + [1] for point in stars] + \
              [point + [2] for point in triangles]

    random.shuffle(dataset)

    write_dataset(dataset, "assignment3/datasets/artificial.csv")


def evaluate(model, dataset, ratio=0.8, num_realizations=20):
    normalizer = Normalizer()
    normalizer.fit(dataset)
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
        plt.xlabel("Épocas")
        plt.ylabel("Soma dos erros")
        plt.show()

    # Plot decision surface
    # if plotting_available:
    #     # Set models with the "mean weights"
    #     model.weights = avg_realization.weights
    #     plot_regression_surface(model,
    #                             normalizer,
    #                             avg_realization.training_set + avg_realization.test_set,
    #                             x_label="X", y_label="Y",
    #                             scatter_label='Base de dados',
    #                             model_label='Predição do modelo', title='Artificial I')


# Generate artificial dataset
# generate_artificial_dataset()

dataset = Dataset("assignment3/datasets/breast-cancer.csv")

model = PerceptronNetwork(learning_rate=0.01, epochs=100, early_stopping=True, verbose=False)
evaluate(model, dataset.load(), ratio=0.8, num_realizations=20)

print("Done!")
