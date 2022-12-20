import torch


args_i1 = "C:\\Users\\nandc\\Dropbox\\research\\A Systems Neuroscience Approach to Continual Learning\\data\\raw\\mnist_train.pt"
args_i2 = "C:\\Users\\nandc\\Dropbox\\research\\A Systems Neuroscience Approach to Continual Learning\\data\\raw\\mnist_test.pt"
args_o = "mnist_permutations.pt"
args_n_tasks = 5
args_seed = 0

torch.manual_seed(args_seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load(args_i1)
x_te, y_te = torch.load(args_i2)
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0
y_tr = y_tr.view(-1).long()
y_te = y_te.view(-1).long()

for t in range(args_n_tasks):
    p = torch.randperm(x_tr.size(1)).long().view(-1)

    tasks_tr.append(['random permutation', x_tr.index_select(1, p), y_tr])
    tasks_te.append(['random permutation', x_te.index_select(1, p), y_te])

torch.save([tasks_tr, tasks_te], args_o)