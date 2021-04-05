import math
from csv_helper import load_dataset, train_test_split

# TODO: Requirements

class KNN:
    def __init__(self):
        self.training_set = None
        self.k = 1

    def train(self, training_set, k):
        self.training_set = training_set
        self.k = k

    def euclidean_distance(self, x, y):
        summation = 0
        count = min(len(x), len(y))
        for i in range(count):
            summation += (x[i] - y[i]) ** 2
        return math.sqrt(summation)

    def get_neighbors(self, point):
        distances = []
        for i in range(len(self.training_set)):
            distance = self.euclidean_distance(point, self.training_set[i][:-1])
            distances.append((i, distance))
        distances.sort(key=lambda x: x[1])
        indexes = list(map(lambda x: x[0], distances[:self.k]))
        result = list(map(lambda index: training_set[index], indexes))
        return result

    def max_class(self, neighbors):
        count = dict()
        for row in neighbors:
            klass = row[-1]
            if klass not in count:
                count[klass] = 0
            count[klass] += 1
        return max(count, key=lambda key: count[key])

    def predict(self, point):
        neighbors = self.get_neighbors(point)
        prediction = self.max_class(neighbors)
        return prediction


filename = 'iris.csv'
dataset = load_dataset(filename)
training_set, test_set = train_test_split(dataset, ratio=0.7, shuffle=True)

# training_set = [
#     [1, 2, 0], # 5
#     [3, 4, 1], # 2.23
#     [5, 6, 2], # 1
#     [7, 8, 0], # 3.6
#     [9, 10, 1], # 6.4
#     [11, 12, 2] # 9.21
# ]

# 2 1 3 0 4 5


k = 11
knn = KNN()
knn.train(training_set, k)


correct_predictions = 0
runs = 1
total_accuracy = 0

for i in range(0, runs):
    for test_input in test_set:
        klass = test_input[-1]
        test_input[-1] = None
        predicted = knn.predict(test_input)
        if predicted == klass:
            correct_predictions += 1
    total_accuracy += correct_predictions / len(test_set)

mean_accuracy = total_accuracy / runs
print(mean_accuracy)