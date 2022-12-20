import torch


args_i = 'raw/cifar100.pt'
args_n_tasks = 20
args_seed  = 0
args_o = 'cifar100.pt'

torch.manual_seed(args_seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(args_i)
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

cpt = int(100 / args_n_tasks) #classes per task

for t in range(args_n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

torch.save([tasks_tr, tasks_te], args_o)