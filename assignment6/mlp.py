import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        dataset = torch.tensor(dataset)
        self.x = dataset[:, :-1]
        self.y = dataset[:, -1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_output)
        )

    def forward(self, x):
        return self.layers(x)


class MLPTorch:
    def __init__(self, num_hidden=2, batch_size=10, regression=False, learning_rate=0.01, epochs=50, verbose=False):
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.regression = regression
        self.model = None
        self.errors = None

    def train(self, training_set):
        if self.regression:
            num_output = 1
        else:
            num_output = int(torch.max(torch.tensor(training_set)[:, -1]).item() + 1)

        dataset = TorchDataset(dataset=training_set)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MLP(num_input=len(training_set[0]) - 1, num_hidden=self.num_hidden, num_output=num_output)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        if self.regression:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        self.errors = list()

        for epoch in range(1, self.epochs + 1):
            loss_sum = 0
            for x, y in dataloader:
                pred = self.model(x)

                if self.regression:
                    loss = loss_fn(pred, y.float().view(-1, 1))
                else:
                    loss = loss_fn(pred, y.long())
                loss_sum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.verbose:
                print("Epoch={:d}  Loss={:.6f}".format(epoch, loss_sum))
            self.errors.append(loss_sum)

    def predict(self, x):
        self.model.eval()
        pred = self.model(torch.tensor(x).float())
        if self.regression:
            return pred.item()
        prob = F.softmax(pred, dim=0)
        return torch.argmax(prob).item()
