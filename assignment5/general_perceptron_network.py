import random
from assignment5.helpers import shape, matrix_product, matrix_sub, matrix_elementwise_product, matrix_t
from assignment5.helpers import matrix_scalar_product, matrix_add


class GeneralPerceptronNetwork:
    def __init__(self, activation_function, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        self.activation_function = activation_function
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

        result = list()
        for row in rows:
            klass = row[-1]
            encoding = self.one_hot_encodings[klass]
            row_copy = row.copy()
            row_copy.pop()
            for i in encoding:
                row_copy.append(i)
            result.append(row_copy)

        return result

    def generate_weights(self):
        assert self.c is not None
        assert self.p is not None

        self.weights = list()
        for i in range(self.p):
            self.weights.append([random.uniform(0, 1) for _ in range(self.c)])

        assert shape(self.weights) == (self.p, self.c)  # Shape: (p, c)

    def train_predict(self, row):
        u_t = matrix_product([row], self.weights)[0]  # Shape: (1, c)
        y_t = [self.activation_function.activate(u) for u in u_t]
        return y_t

    def predict(self, row):
        u_t = self.train_predict(row + [-1])
        y_t = [self.activation_function.step(prob) for prob in u_t]

        count = 0
        for y in y_t:
            count += y

        # In "doubt" area
        if count != 1:
            i_max = 0
            for i in range(len(u_t)):
                if u_t[i] > u_t[i_max]:
                    i_max = i
            for i in range(len(u_t)):
                if i == i_max:
                    y_t[i] = 1
                else:
                    y_t[i] = 0

        for i in range(len(y_t)):
            if y_t[i] == 1:
                return i
        assert False, "Bad one hot encoding"

    def optimize_weights(self, x_t, d_t, y_t):
        e_t = matrix_sub(d_t, y_t)
        y_t_derivative = [[self.activation_function.derivative(x) for x in row] for row in y_t]
        ey = matrix_elementwise_product(e_t, y_t_derivative)  # Shape: (1, c)

        x = matrix_t(x_t)  # Shape: (p, 1)

        xey = matrix_product(x, ey)  # Shape: (p, c)
        nxey = matrix_scalar_product(self.learning_rate, xey)
        self.weights = matrix_add(self.weights, nxey)

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_one_hot_encodings(training_set)
        one_hot_training_set = self.one_hot_encode(training_set)
        self.p = len(one_hot_training_set[0]) - self.c + 1

        self.generate_weights()

        self.errors = list()

        for epoch in range(self.epochs):
            error_sum = 0
            random.shuffle(one_hot_training_set)

            for row in one_hot_training_set:
                # Get the features with bias
                x_t = [row[:-self.c] + [-1]]          # Shape: (1, p)

                # Get desired output
                d_t = [row[-self.c:]]                 # Shape: (1, c)

                # Avoid numerical issues by using values close to 1
                d_t = [[self.activation_function.transform_d(d) for d in row] for row in d_t]

                # Make prediction in one hot format
                y_t = [self.train_predict(x_t[0])]  # Shape: (1, c)

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


