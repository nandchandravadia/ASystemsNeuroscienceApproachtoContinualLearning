import matplotlib.pyplot as plt
import torch

n_tasks = range(5)

# load 'Single' model results
accuracy_single, memory_loss_single = torch.load("C:\\Users\\nandc\Dropbox\\research\\A Systems Neuroscience Approach to Continual Learning\\results\\single.pt")

# load 'Fly' model results
accuracy_fly, memory_loss_fly = torch.load("C:\\Users\\nandc\Dropbox\\research\\A Systems Neuroscience Approach to Continual Learning\\results\\fly.pt")



fig, axes = plt.subplots(1, 2, figsize = (15, 10))



axes[0].plot(n_tasks, accuracy_single, marker = '*', color  = 'blue', linestyle='dashed', alpha=0.7, markersize = 10,
             label = "Vanilla")
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Accuracy (in %)')
axes[0].grid(True)

axes[0].plot(n_tasks, accuracy_fly, marker = '*', color  = 'red', linestyle='dashed', alpha=0.7, markersize = 10,
             label = "FlyModel")
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Accuracy (in %)')
axes[0].grid(True)
axes[0].set_ylim(0,100)




axes[1].plot(n_tasks, memory_loss_single, marker = '*', color  = 'blue', linestyle='dashed', alpha=0.7, markersize = 10,
             label = "Vanilla")
axes[1].set_title('Memory Loss')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Memory Loss')
axes[1].grid(True)

axes[1].plot(n_tasks, memory_loss_fly, marker = '*', color  = 'red', linestyle='dashed', alpha=0.7, markersize = 10,
             label = "FlyModel")
axes[1].set_title('Memory Loss')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Memory Loss')
axes[1].grid(True)

axes[1].set_ylim(0,1)




#fig.savefig('./figs/percentOfNeurons_vs_alpha.png',format='png')


fig.tight_layout()

axes[1].legend()

plt.show()



