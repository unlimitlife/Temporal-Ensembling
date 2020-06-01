import numpy as np
import torch.nn as nn
import torch
import utils


def test(net, test_loader, config, data_config):
    device = config['device']
    task_path = config['save_path']
    num_classes = data_config['total_classes']

    classes = data_config['classes']

    log = utils.Log(task_path)

    net.eval()
    total, single_total_correct, test_loss = 0.0, 0.0, 0.0

    class_per_sample = np.zeros([num_classes])

    single_class_correct = np.zeros([num_classes])
    single_class_accuracy = np.zeros([num_classes])

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            cur_batch_size = images.size(0)
            outputs = net(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * cur_batch_size

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for idx in range(c.size()[0]):
                label = labels[idx]
                single_class_correct[label] += c[idx].item()
                class_per_sample[label] += 1

    range_class = range(num_classes)
    for idx in range_class:
        total += class_per_sample[idx]

        single_total_correct += single_class_correct[idx]
        single_class_accuracy[idx] = 100.0 * single_class_correct[idx] / class_per_sample[idx]

        log.info('Accuracy of %s - single-head: %.3lf%%' %
                    (classes[idx], single_class_accuracy[idx]))

    test_loss /= total
    single_total_accuracy = 100.0 * single_total_correct / total
    log.info("single-head test accuracy: %.3lf%% test_loss: %.3lf test_sample: %d" %
            (single_total_accuracy, test_loss, total))
    return test_loss, single_total_accuracy, single_class_accuracy