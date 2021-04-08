from knn import KNN
from helpers.csv_helper import load_dataset, train_test_split
from helpers.plot_helper import plot_decision_surface

# TODO: Requirements

# training_set = [
#     [1, 2, 0], # 5
#     [3, 4, 1], # 2.23
#     [5, 6, 2], # 1
#     [7, 8, 0], # 3.6
#     [9, 10, 1], # 6.4
#     [11, 12, 2] # 9.21
# ]


def encoded_classes(dataset, encoding):
    # For values not contained in the encoding dict, we use the value max + 1
    unknown = max(encoding.values()) + 1
    for row in dataset:
        klass = row[-1]  # Last element of the array is the class
        if encoding.get(klass) is not None:
            row[-1] = encoding[row[-1]]
        else:
            row[-1] = unknown
    return dataset


def evaluate(get_model, dataset, ratio=0.8, rounds=1):
    total_accuracy = 0

    for i in range(0, rounds):
        correct_predictions = 0

        # Construct the model
        training_set, test_set = train_test_split(dataset, ratio, shuffle=True)
        model = get_model(training_set)

        for row in test_set:
            klass = row[-1]
            predicted = model.predict(row)
            if predicted == klass:
                correct_predictions += 1
        total_accuracy += correct_predictions / len(test_set)

    average_accuracy = total_accuracy / rounds
    print("Accuracy: {:.2f}".format(average_accuracy))


def plot_evaluate(get_model, dataset, ratio=0.8):
    training_set, test_set = train_test_split(dataset, ratio, shuffle=True)
    model = get_model(training_set)
    plot_decision_surface(model, test_set, offset=0.2)
    print('Done!')


def knn_model(k, training_set):
    model = KNN()
    model.train(training_set, k)
    return model


# Load dataset
filename = 'iris.csv'
dataset = load_dataset(filename)

# Encode class value from string to integer
encoding = {'Iris-setosa': 0}
dataset = encoded_classes(dataset, encoding)

# Params
train_test_ratio = 0.8
evaluation_rounds = 10

# Lambda function that constructs the model
get_model = lambda training_set: knn_model(5, training_set)

# Calculate average accuracy for the model
evaluate(get_model, dataset, ratio=train_test_ratio, rounds=evaluation_rounds)

# Plot decision surface
plot_evaluate(get_model, dataset, ratio=train_test_ratio)
