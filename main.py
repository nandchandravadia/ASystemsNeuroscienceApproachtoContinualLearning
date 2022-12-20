import importlib
import random
import uuid
import os
import numpy as np
import torch


def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)

    #only for FlyModel (testing)

    if args.model == "FlyModel":
        N = 250
        for i, (t, x, y) in enumerate(d_tr):
            d_tr[i][1] = x[0:N, :]
            d_tr[i][2] = y[0:N]

        for i, (t,x,y) in enumerate(d_te):
            d_te[i][1] = x[0:N, :]
            d_te[i][2] = y[0:N]

    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:

    def __init__(self, data, args):
        self.data = data

        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        for t in range(n_tasks):

            N = data[t][1].size(0)  # the number of examples per each episode

            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)

            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []

        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]


class Evaluation:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracy = {}
        self.task_accuracy = {}
        self.memory_loss = {}


    def compute_accuracy(self, model, task_ID, x_te):

        current_task = 0

        correct = 0
        total = 0

        with torch.no_grad():
            while current_task <= task_ID:
                test_data = x_te[current_task][1]
                labels = x_te[current_task][2]

                #let's get the predictions from the model
                outputs = model(test_data)

                values, indices = torch.max(outputs, 1)

                correct += (indices == labels).sum().item()
                total += labels.size(0)

                current_task += 1

        self.accuracy[task_ID] = correct / total


        return

    def compute_task_accuracy(self, model, task_ID, x_te):

        correct = 0
        total = 0

        with torch.no_grad():

            test_data = x_te[task_ID][1]
            labels = x_te[task_ID][2]

            # let's get the predictions from the model
            outputs = model(test_data)

            values, indices = torch.max(outputs, 1)

            correct += (indices == labels).sum().item()
            total += labels.size(0)

        self.task_accuracy[task_ID] = correct / total

        return

    def compute_memory_loss(self, model, x_te):

        acc = {}
        num_tasks  = len(x_te)

        with torch.no_grad():
            for task in range(num_tasks):

                correct, total = 0, 0
                test_data = x_te[task][1]
                labels = x_te[task][2]

                # let's get the predictions from the model
                outputs = model(test_data)

                values, indices = torch.max(outputs, 1)

                correct += (indices == labels).sum().item()
                total += labels.size(0)

                self.memory_loss[task] = self.task_accuracy[task] - (correct/total)

        return



def life_experience(model, continuum, x_te, args):

    num_tasks =len(x_te)
    evaluation = Evaluation(num_tasks)

    current_task = 0
    for i, data in enumerate(continuum):

        # First, we need to train on each episode (individually)
        input, task_ID, labels = data



        #keep training batches until we reach a new Episode
        if current_task != task_ID:
            # we reached a new episode (let's test)

            # ======== Evaluation ================ #

            #First, we compute (general) accuracy
            evaluation.compute_accuracy(model = model, task_ID = current_task,
                                        x_te = x_te)

            #Then, we compute (task) accuracy
            evaluation.compute_task_accuracy(model = model, task_ID = current_task,
                                             x_te = x_te)

            # ========== end of evaluation ===========

            current_task += 1

        else:
            # we are in the current episode (keep training)


            loss = model.observe(input, labels, task_ID)
            print("task_ID: {}, batch num: {}, loss: {}".format(task_ID, i, loss))

    # ======== Evaluation (last task) ================ #

    # First, we compute accuracy
    evaluation.compute_accuracy(model=model, task_ID=current_task,
                                        x_te=x_te)

    # Then, we compute (task) accuracy
    evaluation.compute_task_accuracy(model=model, task_ID= current_task,
                                     x_te=x_te)

    # Next, we compute memory loss
    evaluation.compute_memory_loss(model = model, x_te = x_te)


    # ===== end of evaluation ============== #

    return evaluation


class Args:
    def __init__(self, args):

        # model parameters
        self.model = args['model']
        self.n_hiddens = args['n_hiddens']
        self.n_layers = args['n_layers']

        # memory parameters
        self.n_memories = args['n_memories']
        self.memory_strength = args['memory_strength']
        self.finetune = args['finetune']

        # optimizer parameters
        self.n_epochs = args['n_epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']

        # experiment parameters
        self.cuda = args['cuda']
        self.seed = args['seed']
        self.log_every = args['log_every']
        self.save_path = args['save_path']

        # data parameters
        self.data_path = args['data_path']
        self.data_file = args['data_file']
        self.samples_per_task = args['samples_per_task']
        self.shuffle_tasks = args['shuffle_tasks']



if __name__ == "__main__":

    args = {}

    # model parameters
    model = "FlyModel" #"FlyModel"
    n_hiddens = 100
    n_layers = 2

    args['model'] = model
    args['n_hiddens'] = n_hiddens
    args['n_layers'] = n_layers


    # memory parameters
    n_memories = 0
    memory_strength = 0
    finetune = "no"

    args['n_memories'] = n_memories
    args['memory_strength'] = memory_strength
    args['finetune'] = finetune


    # optimizer parameters
    n_epochs = 1
    batch_size = 64 #10
    lr = 0.1

    args['n_epochs'] = n_epochs
    args['batch_size'] = batch_size
    args['lr'] = lr


    # experiment parameters
    cuda = "no"
    seed = 0
    log_every = 100
    save_path = 'results/'

    args['cuda'] = cuda
    args['seed'] = seed
    args['log_every'] = log_every
    args['save_path'] = save_path


    # data parameters
    data_path = "data/"
    data_file = 'mnist.pt'
    samples_per_task = 12500
    shuffle_tasks = "no"

    args['data_path'] = data_path
    args['data_file'] = data_file
    args['samples_per_task'] = samples_per_task
    args['shuffle_tasks'] = shuffle_tasks


    args = Args(args)

    if args.cuda == "yes":
        args.cuda = True
    else:
        args.cuda = False

    if args.finetune == "yes":
        args.finetune = True
    else:
        args.finetune = False


    # multimodal model has one extra layer
    if args.model == 'multimodal':
        args.n_layers -= 1

    # unique identifier
    uid = uuid.uuid4().hex


    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)


    # set up continuum
    continuum = Continuum(x_tr, args)

    dimensionality_expansion = 40
    input_size = 84
    output_size = 10
    sizes = [input_size, dimensionality_expansion * input_size, output_size]

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(args, sizes)
    if args.cuda:
        model.cuda()

    # run model on continuum
    evaluation = life_experience(model, continuum, x_te, args)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


