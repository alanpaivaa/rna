import random
from assignment6.layer import Layer
from assignment6.helpers import shape
from assignment6.activation_functions import LogisticActivationFunction, LinearActivationFunction


class MultiLayerPerceptron:
    def __init__(self, num_hidden=2, regression=False, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.num_hidden = num_hidden
        self.regression = regression
        self.num_inputs = None
        self.num_classes = None
        self.one_hot_encodings = None
        self.layers = None
        self.one_hot_encodings = None
        self.errors = None

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def generate_one_hot_encodings(self, dataset):
        max_class = -1
        for row in dataset:
            max_class = max(max_class, row[-1])
        self.num_classes = max_class + 1

        self.one_hot_encodings = dict()
        for i in range(self.num_classes):
            encoding = [0] * self.num_classes
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

    def initialize_layers(self):
        self.layers = list()
        self.layers.append(Layer(activation_function=LogisticActivationFunction(),
                                 regression=self.regression,
                                 num_inputs=self.num_inputs,
                                 num_neurons=self.num_hidden,
                                 learning_rate=self.learning_rate))
        if self.regression:
            self.layers.append(Layer(activation_function=LogisticActivationFunction(),
                                     regression=self.regression,
                                     num_inputs=self.num_hidden,
                                     num_neurons=self.num_hidden,
                                     learning_rate=self.learning_rate))
            self.layers.append(Layer(activation_function=LinearActivationFunction(),
                                     regression=self.regression,
                                     num_inputs=self.num_hidden,
                                     num_neurons=1,
                                     learning_rate=self.learning_rate))
        else:
            self.layers.append(Layer(activation_function=LogisticActivationFunction(),
                                     regression=self.regression,
                                     num_inputs=self.num_hidden,
                                     num_neurons=self.num_classes,
                                     learning_rate=self.learning_rate))

    def predict(self, row):
        y_t = [row + [1]]  # Shape: (1, num_inputs)
        assert shape(y_t) == (1, self.num_inputs + 1)

        for i in range(len(self.layers)):
            step_result = i == len(self.layers) - 1 and not self.regression
            y_t = self.layers[i].forward(y_t, training=False, step=step_result)
            y_t = [y_t[0] + [1]]  # 1 for bias

        # Remove bias in the end
        y_t = y_t[0][:-1]

        if self.regression:
            return y_t[0]
        else:
            for i in range(len(y_t)):
                if y_t[i] != 0:
                    return i

        assert False, "Wrong result"

    def forward(self, row):
        # Get the features with bias
        x_t = [row[:-self.num_classes] + [1]]  # Shape: (1, num_inputs)
        assert shape(x_t) == (1, self.num_inputs + 1)  # 1 for bias

        for layer in self.layers:
            layer.forward(x_t, training=True)
            x_t = [layer.output[0] + [1]]  # 1 for bias

    def backward(self, row):
        d_t = [row[-self.num_classes:]]         # Shape: (1, num_classes)

        # Update last layer's weights
        self.layers[-1].update_errors(d_t)

        i = len(self.layers) - 2
        while i >= 0:
            self.layers[i].update_errors_hidden(self.layers[i + 1])
            i -= 1

        i = len(self.layers) - 1
        while i >= 0:
            if i > 0:
                x_t = self.layers[i - 1].output
            else:
                x_t = [row[:-self.num_classes]]

            x_t = [x_t[0] + [1]]
            self.layers[i].backward(x_t)
            i -= 1

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        if self.regression:
            self.num_classes = 1
            self.num_inputs = len(training_set[0]) - 1
            one_hot_training_set = training_set
        else:
            self.generate_one_hot_encodings(training_set)
            one_hot_training_set = self.one_hot_encode(training_set)
            self.num_inputs = len(one_hot_training_set[0]) - self.num_classes

        self.initialize_layers()
        self.errors = list()

        for epoch in range(self.epochs):
            error_sum = 0
            random.shuffle(one_hot_training_set)

            for row in one_hot_training_set:
                self.forward(row)
                self.backward(row)

                # Update error summation
                for layer in self.layers:
                    for error in layer.errors[0]:
                        try:
                            error_sum += error ** 2
                        except OverflowError:
                            error_sum = float('inf')

                for layer in self.layers:
                    layer.output = None
                    layer.errors = None

            self.log("epoch {}, error: {:.2f}".format(epoch, error_sum))
            self.errors.append(error_sum)

            # No errors in the epoch, no need to continue training
            if self.early_stopping and error_sum == 0:
                self.log("Early stopping training in epoch {} as no mistakes were made".format(epoch))
                break
