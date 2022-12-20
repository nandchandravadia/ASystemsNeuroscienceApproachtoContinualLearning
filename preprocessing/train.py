import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor


## ============= Save Model ============================= #
save = True

#  =========== Set up parameters ========================
n_epochs = 1
batch_size_train = 64
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10
# =======================================================

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# ==========  Load KMNIST Data =========================
training_data = datasets.KMNIST(root = "data", train = True, download = True,
                                 transform = ToTensor())



testing_data = datasets.KMNIST(root = "data", train = False, download = True,
                                 transform = ToTensor())


train_loader = torch.utils.data.DataLoader(training_data,
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(testing_data,
  batch_size=batch_size_test, shuffle=True)


#Load our model
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), (2, 2))
        x3 = x2.view(-1, self.num_flat_features(x2))
        x4 = F.relu(self.fc1(x3))
        x5 = F.relu(self.fc2(x4))
        x6 = self.fc3(x5)

        return x6, x5

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet()

#Define our optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9)

#Train the Network
for epoch in range(n_epochs):
    running_loss = 0

    for batch_num, data in enumerate(train_loader):
        # get the inputs + labels
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, representations = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


PATH = "LeNet5.pth"

if save:
    torch.save(net.state_dict(), PATH)


sensory_model = LeNet()
sensory_model.load_state_dict(torch.load(PATH))

sensory_model.eval()














