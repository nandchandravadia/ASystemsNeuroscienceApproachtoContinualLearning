import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, args, sizes):
        super(Net, self).__init__()

        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        # setup optimizer + losses
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.001)
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x):
        output = self.net(x)
        return output

    def observe(self, x, y, t):

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(x)
        loss = self.criterion(outputs, y)
        loss.backward()

        self.optimizer.step()

        return loss.item()

