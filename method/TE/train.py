import numpy as np
import selector
import os
import math
import copy
import torch.utils.data
import utils
from dataset import taskset
from torchvision.datasets import ImageFolder
import torch.nn as nn
from .test import test
from .loss import temporal_loss
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


def toImage(x):
    return Image.fromarray(np.uint8(np.transpose(x,(1,2,0))))
    

def rampup(epoch, method_config):
    rampup_length = method_config['rampup_length']
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch, method_config):
    rampdown_length = method_config['rampdown_length']
    num_epochs = method_config['num_epochs']
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0

def train(config, method_config, data_config, logger):
    """
    Args:
        config (dict): config file dictionary.
            model (str): name of network. [selector.model(model, ...)]
            classes_per_task (int): classes per task.
            DA (bool): if True, apply data augment.
            memory_cap (int): sample memory size.
            num_workers (int): how many subprocesses to use for data loading.
                               0 means that the data will be loaded in the main process. (default: 0)
            batch_size (int): how many samples per batch to load. (default: 1)
            device (torch.device): gpu or cpu.
            data_path (str): root directory of dataset.
            save_path (str): directory for save. (not taskwise)
        data_config (dict): data config file dictionary.
            dataset (str): name of dataset.
            total_classes (int): total class number of dataset.
            curriculums (list): curriculum list.
            classes (list): class name list.
        method_config (dict): method config file dictionary.
            method (str): name of method.
            process_list (list): process list.
            package (string): current package name.
        logger (Logger): logger for the tensorboard.
    """
    model = config['model']
    device = config['device']
    data_path = config['data_path']
    save_path = config['save_path']

    num_classes = data_config['total_classes']
    dataset = data_config['dataset']
    train_transform = data_config['transform']['train']
    test_transform = data_config['transform']['test']

    num_labels = method_config['temporal_ensemble']['num_labels']

    test_task = []
    train_task = []
    
    train_taskset = taskset.Taskset(data_path, train=True, transform=train_transform, num_labels=num_labels, num_classes=num_classes)
    test_taskset = taskset.Taskset(data_path, train=False, transform=test_transform)

    '''Make network'''
    net = selector.model(model, device, num_classes)
        
    _train(net, train_taskset, test_taskset, config, method_config, data_config, logger)

def _train(net, train_taskset, test_taskset, config, method_config, data_config, logger):
    """
    Args:
        config (dict): config file dictionary.
            task_path (str): directory for save. (taskwise, save_path + task**)
            cml_classes (int): size of cumulative taskset
    """
    batch_size = config['batch_size']
    save_path = config['save_path']
    num_workers = config['num_workers']
    device = config['device']
    num_classes = data_config['total_classes']

    process_list = method_config['process_list']

    method_config = method_config['temporal_ensemble']
    epochs = method_config['num_epochs']
    num_labels = method_config['num_labels']
    lr_max = method_config['learning_rate_max']
    rampdown_beta1_target = method_config['rampdown_beta1_target']
    adam_beta1 = method_config['adam_beta1']
    adam_beta2 = method_config['adam_beta2']
    adam_epsilon = method_config['adam_epsilon']
    alpha = method_config['alpha']
    std = method_config['std']
    unsup_weight_max = method_config['unsup_weight_max']
    augment_translation = method_config['augment_translation']
    augment_mirror = method_config['augment_mirror']
    
    #ramp_up_mult = method_config['ramp_up_mult']

    ###
    DATA_NO_LABEL=-1
    loss_ce = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL).cuda()
    loss_tempens = torch.nn.MSELoss(reduction='mean').cuda()
    ###

    log = utils.Log(save_path)
    epoch = 0
    single_best_accuracy, multi_best_accuracy = 0.0, 0.0

    for process in process_list:        

        if num_labels != 'all':
            unsup_weight_max *= 1.0 * num_labels / len(train_taskset)

        log.info("Start Training")

        losses = []
        sup_losses = []
        unsup_losses = []
        best_loss = 20.
        Z = torch.zeros((len(train_taskset), num_classes)).to(device)
        z_tilda = torch.zeros((len(train_taskset), num_classes)).to(device)

        train_loader = torch.utils.data.DataLoader(train_taskset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
        for ep in range(epochs):
            log.info("%d Epoch Started" % epoch)
            net.train()
            epoch_loss = 0.0
            total = 0
        
            rampup_value = rampup(ep, method_config)
            rampdown_value = rampdown(ep, method_config)
            lr = rampup_value * rampdown_value * lr_max
            adam_beta1 = rampdown_value * adam_beta1 + (1.0 - rampdown_value) * rampdown_beta1_target
            unsup_weight = rampup_value * unsup_weight_max
            if ep == 0:
                unsup_weight = 0.0

            optimizer = selector.optimizer.RobustAdam(lr=lr, betas=(adam_beta1, adam_beta2))(net.parameters())
            
            unsup_weight = torch.autograd.Variable(torch.FloatTensor([unsup_weight]).cuda(), requires_grad=False)
            epoch_predictions = torch.zeros((len(train_taskset), num_classes)).to(device)
        
            l = []
            supl = []
            unsupl = []
            if ep == 0:
                for i, data in enumerate(train_loader):
                    images = data[0].to(device)
                    with torch.no_grad():
                        net(images, moving_average=True, init_mode=True)
                        break
                log.info("\ninit mode finish")
            
            for i, data in enumerate(train_loader):
                utils.printProgressBar(i + 1, len(train_loader), prefix='train')
                images, labels, idx = data[0].to(device), data[1].to(device), data[2].to(device)
                cur_batch_size = images.size(0)
                optimizer.zero_grad()
                outs = net(images)
                ####zcomp = Variable(z_tilda[idx], requires_grad=False)

                #loss, suploss, unsuploss, nbsup = temporal_loss(outs, zcomp, unsup_weight, labels)
                #outputs[idx.view(cur_batch_size)] = outs.data.clone()
                #l.append(loss.item())
                #supl.append(nbsup * suploss.item())
                #unsupl.append(unsuploss.item())
                ###############
                
                tempens_target_var = torch.autograd.Variable(z_tilda[idx].cuda(), requires_grad=False)
                predictions = F.softmax(outs, dim=1)

                suploss = loss_ce(outs, labels) / cur_batch_size
                unsuploss = unsup_weight * loss_tempens(predictions, tempens_target_var)

                nbsup = labels.data.ne(DATA_NO_LABEL).sum().float()
                loss = suploss + unsuploss

                l.append(loss.item())
                supl.append(suploss.item())
                unsupl.append(unsuploss.item())
                
                ################
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * cur_batch_size
                total += cur_batch_size
                
                for i, j in enumerate(idx):
                    epoch_predictions[j] = predictions[i].data.clone()

            #epoch_loss /= total

            #if ep == (epochs - 1):
            if (ep+1) % 6 == 0:
                test_loss, single_total_accuracy, single_class_accuracy = \
                    test(net, test_loader, config, data_config)
                
            # update temporal ensemble
            
            #Z = (alpha * Z) + (1.0 - alpha) * epoch_predictions
            #self.training_targets = self.ensemble_prediction / (1.0 - self.args.prediction_decay ** (
            #        (epoch - self.start_epoch) + 1.0))
            Z = alpha * Z + (1. - alpha) * epoch_predictions
            z_tilda = Z * (1. / (1. - alpha ** (epoch + 1)))
            del epoch_predictions
            # handle metrics, losses, etc.
            eloss = np.mean(l)
            losses.append(eloss)
            sup_losses.append((1. / num_labels) * np.sum(supl))  
            unsup_losses.append(np.mean(unsupl))
                
            log.info("epoch: %d  train_loss: %.3lf  train_sample: %d" % (epoch, losses[-1], total))
            log.info("sup loss : "+str(sup_losses[-1]))
            log.info("unsup loss : "+str(unsup_losses[-1]))
            #log.info("sup loss : "+str(supl[-1]))
            #log.info("unsup loss : "+str(unsupl[-1]))
            log.info('unsupervised loss weight : {}'.format(unsup_weight))

            for n,p in net.named_parameters():
                if torch.sum(p != p)>0 or torch.sum(p>100)>0 :
                    print('NAN track!', n, p)

            logger.epoch_step()
            epoch += 1

        log.info("Finish Training")