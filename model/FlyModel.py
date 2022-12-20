import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, args, sizes):
        super(Net, self).__init__()

        #network parameters
        self.d = sizes[0]  # input size
        self.m = sizes[1]  #kenyon cell dims
        self.o = sizes[2]  #MBON dims

        self.sparsity = 0.1

        self.W1 = self.first_layer_weights(m = self.m, d = self.d, sparsity = self.sparsity) #static weights
        self.W2 = torch.zeros(self.o, self.m) #trainable weights

        # ======= init network ==================
        self.layer1 = nn.Linear(self.d, self.m, bias = False)
        with torch.no_grad():
            self.layer1.weight.copy_(self.W1)

        self.layer2 = nn.Linear(self.m, self.o, bias = False)

        with torch.no_grad():
            self.layer2.weight.copy_(self.W2)


    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        return x2

    def observe(self, x, y, t):

        c = self.layer1(x)
        c_hat = self.k_winner_take_all(k = int(.05*self.m), c = c)

        self.learning_rule(c_hat, beta= .01, forgetting = 0, target = y)

        loss = nn.CrossEntropyLoss()
        output = loss(F.softmax(self(x), 1), y)

        return output.item()

    def first_layer_weights(self, m = None, d = None, sparsity = 0.1):
        return torch.from_numpy(generate_patterns(d, m, sparsity))


    def k_winner_take_all(self, k, c):

        values, indices = torch.topk(input = c,  k = k, dim = 1, largest = True, sorted = False)

        c_dim = c.shape[1]

        # inhibit less active KC's (keep only the k-active KC's)
        for i in range(c_dim):
            if c[0, i] not in values:
                c[0,i] = 0

        #apply normalization
        c_hat = self.min_max_normalization(c)

        return c_hat

    def min_max_normalization(self, c_hat):

        c_max = torch.max(c_hat)
        c_min = torch.min(c_hat[c_hat>0])

        c_dim = c_hat.shape[1]

        for i in range(c_dim):
            if c_hat[0,i] != 0:
                c_hat[0,i] = (c_hat[0,i] - c_min)/(c_max - c_min)

        return c_hat


    def learning_rule(self, c, beta, forgetting, target):

        memory_decay = 1 - forgetting

        output_values, KC_dim =  self.layer2.weight.shape
        target_value = int(target[0])

        for o in range(output_values):
            if o == target_value: #update *target* weights
                for i in range(KC_dim):
                    with torch.no_grad():
                        self.layer2.weight[o, i] = memory_decay * self.layer2.weight[o,i] + beta*c[0,i]

            else: #partial freezing of synapses
                if memory_decay != 1:
                    for i in range(KC_dim):
                        with torch.no_grad():
                            self.layer2.weight[o, i] = memory_decay * self.layer2.weight[o, i]

        return

    def compute_cross_entropy_loss(self):
        return None



def generate_patterns(n, dim, sparsity):

    patterns = np.zeros(shape = (dim, n))  # each column is a pattern

    for index in range(0, n):
        pattern = generate_binary_vector(dim, sparsity)

        patterns[:, index] = pattern

    return patterns


def generate_binary_vector(dim, sparsity):
    # here, sparsity is defined by the number of (1)
    # i.e., more (0), more 'sparse'

    values = [1, 0]

    vec = np.zeros(shape=dim)

    for index, val in enumerate(vec):
        ind = np.random.binomial(1, 1-sparsity)
        vec[index] = values[ind]

    return vec




"""
dimensionality_expansion = 40
input_size = 80
output_size = 10
sizes = [input_size, dimensionality_expansion*input_size, output_size]

net = Net(args= None, sizes = sizes)

batch_size = 1

inputs = torch.rand(batch_size, input_size)

labels = torch.randint(low=0, high = 9, size = (batch_size,))


net.observe(x = inputs, y = labels, t = 1)
"""
