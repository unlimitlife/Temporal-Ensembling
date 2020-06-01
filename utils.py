import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import compress

def printProgressBar(iteration, total, prefix='Progress: ', suffix='Complete', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if iteration > total:
        total = iteration
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


def save_checkpoint(state, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)

# Loads the saved state to be used for training.
def load_checkpoint(model, filepath):
    """
    Args:
        model (nn.Module): model that will load a checkpoint
        filepath (str): the path that checkpoint is on
    """
    state = torch.load(filepath)
    model.load_state_dict(state['state_dict'])
    del state
    torch.cuda.empty_cache()


class Log(object):
    def __init__(self, root):
        if not os.path.isdir(root):
            os.makedirs(root)
        self.root = os.path.join(root, 'log.log')

    def info(self, string):
        print(string)
        with open(self.root, 'a') as f:
            f.write(string + '\n')

class RobustAdam(Optimizer):
    # Convert NaNs to zeros.

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                    weight_decay=weight_decay, amsgrad=amsgrad)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(RobustAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(RobustAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        def clear_nan(x):    
            return torch.where(torch.isnan(x), torch.zeros([1], dtype=torch.float32).to(self.device), x)
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg = clear_nan(torch.add(torch.mul(exp_avg,beta1), grad, alpha=1 - beta1))
                exp_avg_sq = clear_nan(torch.addcmul(torch.mul(exp_avg_sq,beta2), grad, grad, value=1 - beta2))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data = clear_nan(torch.addcdiv(p.data, exp_avg, denom, value=-step_size))

        return loss