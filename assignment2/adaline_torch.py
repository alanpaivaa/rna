import random
import torch
from torch import nn
import torch.nn.functional as F


class AdalineNN(nn.Module):
    def __init__(self, num_input, num_output):
        super(AdalineNN, self).__init__()
        self.linear = nn.Linear(in_features=num_input, out_features=num_output, bias=True)

    def forward(self, x):
        return self.linear(x)


class AdalineTorch:
    def __init__(self, learning_rate=0.01, epochs=50, early_stopping=True, verbose=False):
        super(AdalineTorch, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.model = None
        self.errors = None

    def predict(self, x):
        with torch.no_grad():
            pred = self.model(torch.Tensor(x)).item()
        return pred

    def train(self, training_set):
        # for epoch in epochs
        self.model = AdalineNN(num_input=len(training_set[0]) - 1, num_output=1)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.errors = list()
        for epoch in range(self.epochs):
            error_sum = 0
            random.shuffle(training_set)

            ts = torch.Tensor(training_set)
            x = ts[:, :-1]
            y = ts[:, -1]

            for i in range(x.shape[0]):
                y_hat = self.model(x[i])
                loss = F.mse_loss(y_hat, y[i].view(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                try:
                    error_sum += loss.item() ** 2
                except OverflowError:
                    error_sum = float('inf')

            self.errors.append(error_sum)
