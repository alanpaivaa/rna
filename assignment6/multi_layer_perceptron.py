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
        self.num_hidden = num_hidden
        self.num_inputs = None
        self.num_classes = None
        self.one_hot_encodings = None
        self.layers = None
        self.one_hot_encodings = None

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
        self.layers.append(Layer(activation_function=HyperbolicTangentActivationFunction(), num_inputs=self.num_inputs, num_neurons=self.num_hidden))
        self.layers.append(Layer(activation_function=LinearActivationFunction(), num_inputs=self.num_hidden, num_neurons=self.num_classes))

    def forward(self, row):
        # Get the features with bias
        x_t = [row[:-self.num_classes] + [-1]]  # Shape: (1, num_inputs)
        assert shape(x_t) == (1, self.num_inputs)

        for layer in self.layers:
            layer.forward(x_t)
            x_t = layer.output

    def train(self, training_set):
        assert len(training_set) > 0
        assert len(training_set[0]) > 0

        self.generate_one_hot_encodings(training_set)
        one_hot_training_set = self.one_hot_encode(training_set)
        self.num_inputs = len(one_hot_training_set[0]) - self.num_classes + 1  # 1 for bias

        self.initialize_layers()

        for epoch in range(self.epochs):
            error_sum = 0
            random.shuffle(one_hot_training_set)

            for row in one_hot_training_set:
                self.forward(row)

                # Get desired output
                # d_t = [row[-self.num_classes:]]  # Shape: (1, num_classes)
                # assert shape(d_t) == (1, self.num_classes)
                #
                # Avoid numerical issues by using values close to 1
                # d_t = [[self.activation_function.transform_d(d) for d in row] for row in d_t]
                #
                # # Make prediction in one hot format
                # y_t = [self.train_predict(x_t[0])]  # Shape: (1, c)
                #
                # # Update error summation
                # errors = matrix_sub(d_t, y_t)
                # for error in errors[0]:
                #     try:
                #         error_sum += error ** 2
                #     except OverflowError:
                #         error_sum = float('inf')
                #
                # # Update weights
                # self.optimize_weights(x_t, d_t, y_t)


model = MultiLayerPerceptron(num_hidden=2)
training_set = [
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
]

model.train(training_set)
print("Done!")
