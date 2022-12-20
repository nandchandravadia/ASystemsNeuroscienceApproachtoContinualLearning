import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision import datasets

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

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

testing_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size_train = 60000
batch_size_test = 10000

train_loader = torch.utils.data.DataLoader(training_data,
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(testing_data,
  batch_size=batch_size_test, shuffle=True)



args_o = "mnist.pt"
args_n_tasks = 5
args_seed = 0

torch.manual_seed(args_seed)

tasks_tr = []
tasks_te = []


# load in pre-processing (sensory) module
Sensory_PATH = "C:\\Users\\nandc\\Dropbox\\research\\A Systems Neuroscience Approach to Continual Learning\\preprocessing\\LeNet5.pth"
#Sensory_Model_PATH = "./preprocessing/LeNet5.py"
sensory_model = LeNet()
sensory_model.load_state_dict(torch.load(Sensory_PATH))
sensory_model.eval()

for i, data in enumerate(train_loader):
    ignore, x_tr = sensory_model(data[0].float())
    y_tr = data[1].long()

for i, data in enumerate(test_loader):
    ignore, x_te = sensory_model(data[0].float())
    y_te = data[1].long()


cpt = 2


#get the range of values in MNIST
cpt = 2
loop = np.arange(start = 0, stop = 9, step = cpt)

for t in loop:
    start = t
    stop = t+1

    i_tr = ((y_tr == start) | (y_tr == stop)).nonzero().view(-1)
    i_te = ((y_te == start) | (y_te == stop)).nonzero().view(-1)
    tasks_tr.append([(start, stop), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    tasks_te.append([(start, stop), x_te[i_te].clone(), y_te[i_te].clone()])

torch.save([tasks_tr, tasks_te], args_o)