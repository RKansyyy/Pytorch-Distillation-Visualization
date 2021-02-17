import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import pickle


class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_accuracy(predictions, labels):

    prediction = np.argmax(predictions.detach().cpu().numpy(), axis=1)
    return np.sum(prediction == labels.detach().cpu().numpy())/float(len(labels))


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo)

    return dictionary


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
