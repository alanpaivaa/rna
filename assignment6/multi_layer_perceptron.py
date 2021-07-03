import random
from assignment6.layer import Layer
from assignment6.helpers import shape
from assignment6.activation_functions import HyperbolicTangentActivationFunction, LinearActivationFunction


class MultiLayerPerceptron:
    def __init__(self, num_hidden=2, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.num_hidden = num_hidden + 1  # 1 for bias
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

    # TODO: Make layer creation more generic and parameterized
    def initialize_layers(self):
        self.layers = list()
        self.layers.append(Layer(activation_function=HyperbolicTangentActivationFunction(), num_inputs=self.num_inputs, num_neurons=self.num_hidden, learning_rate=self.learning_rate))
        self.layers.append(Layer(activation_function=LinearActivationFunction(), num_inputs=self.num_hidden, num_neurons=self.num_classes, learning_rate=self.learning_rate))

    def forward(self, row):
        # Get the features with bias
        x_t = [row[:-self.num_classes] + [-1]]  # Shape: (1, num_inputs)
        assert shape(x_t) == (1, self.num_inputs)

        for layer in self.layers:
            layer.forward(x_t)
            x_t = layer.output

    def backward(self, row):
        x_t = [row[:-self.num_classes] + [-1]]  # Shape: (1, num_inputs)
        d_t = [row[-self.num_classes:]]         # Shape: (1, num_classes)

        # Update last layer's weights
        self.layers[-1].backward(self.layers[-2].output, d_t)

        i = len(self.layers) - 2
        while i >= 0:
            self.layers[i].backward_hidden(x_t, self.layers[i + 1])
            i -= 1

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_one_hot_encodings(training_set)
        one_hot_training_set = self.one_hot_encode(training_set)
        self.num_inputs = len(one_hot_training_set[0]) - self.num_classes + 1  # 1 for bias

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

            self.log("epoch {}, error: {:.2f}".format(epoch, error_sum))
            self.errors.append(error_sum)

            # No errors in the epoch, no need to continue training
            if self.early_stopping and error_sum == 0:
                self.log("Early stopping training in epoch {} as no mistakes were made".format(epoch))
                break


model = MultiLayerPerceptron(num_hidden=2, verbose=True)
training_set = [
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 0],
]

model.train(training_set)
print("Done!")
