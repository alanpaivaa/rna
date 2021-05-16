import random
from assignment3.helpers import shape, matrix_product, step, matrix_sub, matrix_t, matrix_scalar_product, matrix_add


class PerceptronNetwork:
    def __init__(self, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.one_hot_encodings = None
        self.c = None  # Number of classes
        self.p = None  # Number of features (including bias)
        self.weights = None
        self.errors = None

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

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

        assert shape(self.weights) == (self.p, self.c)  # Shape: (p, c)

    def predict_one_hot(self, row):
        u_values = matrix_product([row], self.weights)[0]
        y_values = [step(u) for u in u_values]

        count = 0
        for y in y_values:
            count += y

        # Not in "doubt" area
        if count != 1:
            i_max = 0
            for i in range(len(u_values)):
                if u_values[i] > u_values[i_max]:
                    i_max = i
            for i in range(len(u_values)):
                if i == i_max:
                    y_values[i] = 1
                else:
                    y_values[i] = 0

        return y_values

    def predict(self, row):
        y_values = self.predict_one_hot(row)
        for i in range(len(y_values)):
            if y_values[i] == 1:
                return i
        assert False, "Bad one hot encoding"

    def optimize_weights(self, x_t, d_t, y_t):
        e_t = matrix_sub(d_t, y_t)   # Shape: (1, c)
        x = matrix_t(x_t)            # Shape: (p, 1)

        ex = matrix_product(x, e_t)  # Shape: (p, c)
        nex = matrix_scalar_product(self.learning_rate, ex)
        self.weights = matrix_add(self.weights, nex)

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_one_hot_encodings(training_set)
        self.one_hot_encode(training_set)
        self.p = len(training_set[0]) - self.c + 1

        self.generate_weights()

        self.errors = list()

        for epoch in range(self.epochs):
            error_sum = 0
            random.shuffle(training_set)

            for row in training_set:
                # Get the features with bias
                x_t = [row[:-self.c] + [-1]]          # Shape: (1, p)

                # Get desired output
                d_t = [row[-self.c:]]                 # Shape: (1, c)

                # Make prediction in one hot format
                y_t = [self.predict_one_hot(x_t[0])]  # Shape: (1, c)

                # Update error summation
                errors = matrix_sub(d_t, y_t)
                for error in errors[0]:
                    error_sum += error ** 2

                # Update weights
                self.optimize_weights(x_t, d_t, y_t)

            self.log("epoch {}, error: {:.2f}".format(epoch, error_sum))
            self.errors.append(error_sum)

            # No errors in the epoch, no need to continue training
            if self.early_stopping and error_sum == 0:
                self.log("Early stopping training in epoch {} as no mistakes were made".format(epoch))
                break


