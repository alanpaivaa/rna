from helpers.dataset import Dataset
from assignment2.perceptron import Perceptron
from helpers.csv_helper import train_test_split


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

# Iris dataset
iris_encodings = [
    {'Iris-setosa': 0},      # Binary: 0 - Setosa, 1 - Others
    {'Iris-versicolor': 0},  # Binary: 0 - Virginica, 1 - Others
    {'Iris-virginica': 0},   # Binary: 0 - Versicolor, 1 - Others
    {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Multiclass
]
dataset = Dataset('assignment2/datasets/iris.csv', encoding=iris_encodings[1])

ratio = 0.8

model = Perceptron(epochs=50)
evaluate(model, dataset, ratio=ratio)
