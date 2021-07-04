from assignment6.helpers import shape, matrix_product, matrix_sub, matrix_elementwise_product
from assignment6.helpers import matrix_t, matrix_scalar_product, matrix_add
import random


class Layer:
    def __init__(self, activation_function, num_inputs, num_neurons, learning_rate):
        self.activation_function = activation_function
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.errors = None
        self.output = None
        self.weights = [[random.uniform(0, 1) for _ in range(self.num_neurons)] for _ in range(self.num_inputs + 1)]
        assert shape(self.weights) == (self.num_inputs + 1, self.num_neurons)  # 1 for bias

    def step_predict(self, u_t, y):
        y_t = [self.activation_function.step(prob) for prob in y]

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

        return y_t

    def forward(self, x_t, training=False, step=False):
        u_t = matrix_product(x_t, self.weights)[0]  # Shape: (1, num_neurons)
        assert shape([u_t]) == (1, self.num_neurons)

        y_t = [self.activation_function.activate(u) for u in u_t]

        # Some activation functions don't have clear boundaries [0, 1], so we need the step function
        if self.activation_function.step_train() or step:
            y_t = self.step_predict(u_t, y_t)

        if training:
            self.output = [y_t]

        return [y_t]

    def update_errors(self, d_t):
        # Avoid numerical issues by using values close to 1
        # d_t = [[self.activation_function.transform_d(d) for d in row] for row in d_t]  # Shape: (1, num_neurons)
        y_t = self.output  # Shape: (1, num_neurons)
        e_t = matrix_sub(d_t, y_t)  # Shape: (1, num_neurons)
        self.errors = e_t

    def update_errors_hidden(self, next_layer):
        next_activation_function = next_layer.activation_function
        next_weights = next_layer.weights
        next_output = next_layer.output
        next_errors = next_layer.errors

        e_t = list()
        for i in range(self.num_neurons):
            s = 0
            for j in range(next_layer.num_neurons):
                s += next_weights[i][j] * next_activation_function.derivative(next_output[0][j]) * next_errors[0][j]
            e_t.append(s)
        e_t = [e_t]
        assert shape(e_t) == (1, self.num_neurons)
        self.errors = e_t

    def backward(self, x_t):
        e_t = self.errors
        y_t = self.output
        y_t_derivative = [[self.activation_function.derivative(x) for x in row] for row in y_t]
        ey = matrix_elementwise_product(e_t, y_t_derivative)  # Shape: (1, num_neurons)

        x = matrix_t(x_t)  # Shape: (num_inputs, 1)

        xey = matrix_product(x, ey)  # Shape: (num_inputs, num_neurons)
        nxey = matrix_scalar_product(self.learning_rate, xey)

        assert shape(nxey) == shape(self.weights)
        self.weights = matrix_add(self.weights, nxey)
