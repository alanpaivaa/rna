from assignment4.helpers import train_test_split, mean, standard_deviation
from assignment4.dataset import Dataset
from assignment4.general_perceptron import GeneralPerceptron
from assignment4.realization import Realization
from assignment4.scores import Scores
from assignment4.normalizer import Normalizer
from assignment4.activation_functions import LinearActivationFunction, SigmoidLogisticActivationFunction

# Import plotting modules, if they're available
try:
    from assignment4.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


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


# Generate artificial dataset
# generate_artificial_dataset()

dataset = Dataset("assignment4/datasets/artificial.csv")

# activation_function = LinearActivationFunction()
activation_function = SigmoidLogisticActivationFunction()

learning_rate = 0.01
epochs = 300
split_ratio = 0.8
num_realizations = 20

model = GeneralPerceptron(activation_function,
                          learning_rate=learning_rate,
                          epochs=epochs,
                          early_stopping=True,
                          verbose=False)
evaluate(model, dataset.load(), ratio=split_ratio, num_realizations=num_realizations)

print("Done!")
