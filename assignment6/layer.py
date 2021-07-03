from assignment6.helpers import shape, matrix_product
import random


class Layer:
    def __init__(self, activation_function, num_inputs, num_neurons):
        self.activation_function = activation_function
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.error = None
        self.output = None
        self.weights = [[random.uniform(0, 1) for _ in range(self.num_neurons)] for _ in range(self.num_inputs)]
        assert shape(self.weights) == (self.num_inputs, self.num_neurons)

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

    def forward(self, x_t):
        u_t = matrix_product(x_t, self.weights)[0]  # Shape: (1, num_neurons)
        assert shape([u_t]) == (1, self.num_neurons)

        y_t = [self.activation_function.activate(u) for u in u_t]

        # TODO: This probably only makes sense in the last layer
        # Some activation functions don't have clear boundaries [0, 1], so we need the step function
        # if self.activation_function.step_train():
        #     y_t = self.step_predict(u_t, y_t)

        self.output = [y_t]
