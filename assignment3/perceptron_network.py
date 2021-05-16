import random
from assignment3.helpers import shape


class PerceptronNetwork:
    def __init__(self):
        self.one_hot_encodings = None
        self.c = None  # Number of classes
        self.p = None  # Number of features (including bias)
        self.weights = None

    def generate_one_hot_encodings(self, dataset):
        max_class = -1
        for row in dataset:
            max_class = max(max_class, row[-1])
        self.c = max_class + 1

        self.one_hot_encodings = dict()
        for i in range(self.c):
            encoding = [0] * self.c
            encoding[i] = 1
            self.one_hot_encodings[i] = encoding

    def one_hot_encode(self, rows):
        assert self.one_hot_encodings is not None

        for row in rows:
            klass = row[-1]
            encoding = self.one_hot_encodings[klass]
            row.pop()
            for i in encoding:
                row.append(i)

    def generate_weights(self):
        assert self.c is not None
        assert self.p is not None

        self.weights = list()
        for i in range(self.p):
            self.weights.append([random.uniform(0, 1) for _ in range(self.c)])

        assert shape(self.weights) == (self.p, self.c)

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_one_hot_encodings(training_set)
        self.one_hot_encode(training_set)
        self.p = len(training_set[0]) - self.c + 1

        self.generate_weights()
