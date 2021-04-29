import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from torch.autograd import Variable
from StudentModel import *
import numpy as np
import matplotlib.pyplot as plt
from Imagenet64 import Imagenet64
import os


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


def get_topk_accuracy(output, target, k):

    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, False)

    pred = pred.detach().cpu().numpy()
    correct = 0
    for i, label in enumerate(target):
        if label.item() in pred[i]:
            correct += 1

    return correct/batch_size


def get_mean_std(dataloader):
    """
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5
    :param dataloader:
    :return: return mean and standard deveation of a dataset
    """
    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, labels in dataloader:

        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def imshow(img):
    plt.imshow(np.transpose(img[0].numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def get_cifar10_dataloaders(batch_size):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.CIFAR10(root='../data/', train=True, transform=train_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = torchvision.datasets.CIFAR10(root='../data/', train=False, transform=val_transform, download=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def get_vgg19_bn(num_classes, pretrained, distilled):

    teacher_model = models.vgg19_bn(pretrained=False)
    teacher_model.classifier = nn.Sequential(
        nn.Linear(in_features=teacher_model.classifier[0].in_features, out_features=4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 1000),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(1000, num_classes)
    )
    if pretrained:
        if distilled:
            teacher_model.load_state_dict(torch.load(f'../models/vgg19_bn_imagenet_{num_classes}/best_weights_vgg19_distilled_{num_classes}'))
        else:
            teacher_model.load_state_dict(torch.load(f'../models/vgg19_bn_imagenet_{num_classes}/best_weights_vgg19_bn_imagenet_{num_classes}'))

    return teacher_model, f'vgg19_distilled_{num_classes}' if distilled else f'vgg19_bn_imagenet_{num_classes}'


def image_loader(image_name, imsize, device):

    loader = transforms.Compose([transforms.Resize(imsize),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.4358, 0.4584, 0.4358], std=[0.2037, 0.1990, 0.2037])
    ])

    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)


def get_distilled_3l():

    student_model = StudentModel()
    student_model.load_state_dict(torch.load('../models/vgg19_bn/best_distilled_model_weights'))

    return student_model, 'vgg19_bn_dist_3conv'


def get_3conv_2_fc_40(num_classes, pretrained=False, distilled=True):

    student_model = StudentModel(num_classes)
    if pretrained:
        if distilled:
            student_model.load_state_dict(torch.load('../models/vgg19_bn_imagenet_5/best_weights_3_conv_2_fc_40cl'))
        else:
            student_model.load_state_dict(torch.load('../models/3_conv_2_fc_40cl/best_weights_3_conv_2_fc_40cl'))

    return student_model, '3_conv_2_fc_40cl'


def get_5conv_2_fc_10(num_classes, pretrained=False, distilled=True):

    student_model = StudentModel3(num_classes)
    if pretrained:
        if distilled:
            student_model.load_state_dict(torch.load('../models/5_conv_2_fc_10_test/best_weights_5_conv_2_fc_10_test'))
        else:
            student_model.load_state_dict(torch.load('../models/5_conv_2_fc_10_test/best_weights_5_conv_2_fc_10_test'))

    return student_model, '5_conv_distilled' if distilled else '5_conv_2_fc_10_test2'


def get_3_conv_2_fc(num_classes, pretrained=False, distilled=True, flag=True):

    if flag:
        student_model = StudentModel(num_classes)
    else:
        student_model = StudentModel_fc(num_classes)
    if pretrained:
        if distilled:
            student_model.load_state_dict(torch.load(f'../models/vgg19_bn_imagenet_{num_classes}/best_weights_3_conv_2_fc_distilled_{num_classes}'))
        else:
            student_model.load_state_dict(torch.load(f'../models/3_conv_2_fc_{num_classes}/best_weights_3_conv_2_fc_{num_classes}'))

    return student_model, f'3_conv_2_fc_distilled_{num_classes}' if distilled else f'3_conv_2_fc_{num_classes}'


def get_5conv_2_fc_13(num_classes, pretrained=False, distilled=False):

    student_model = StudentModel3(num_classes)
    if pretrained:
        if distilled:
            student_model.load_state_dict(torch.load(f'../models/5_conv_2_fc_{num_classes}/best_weights_5_conv_distilled_{num_classes}'))
        else:
            student_model.load_state_dict(torch.load(f'../models/5_conv_2_fc_{num_classes}/best_weights_5_conv_2_fc_{num_classes}'))

    return student_model, f'5_conv_distilled_{num_classes}' if distilled else f'5_conv_2_fc_{num_classes}'


def get_6_conv_2_fc(num_classes, pretrained=False, distilled=False):

    student_model = StudentModel4(num_classes)
    if pretrained:
        if distilled:
            student_model.load_state_dict(torch.load(f'../models/vgg19_bn_imagenet_{num_classes}/best_weights_6_conv_distilled_{num_classes}'))
        else:
            student_model.load_state_dict(torch.load(f'../models/6_conv_2_fc_{num_classes}/best_weights_6_conv_2_fc_{num_classes}'))

    return student_model, f'6_conv_distilled_{num_classes}' if distilled else f'6_conv_2_fc_{num_classes}'


def get_2conv_2_fc(num_classes, pretrained=False, distilled=True):

    student_model = StudentModel2(num_classes)
    if pretrained:
        if distilled:
            student_model.load_state_dict(torch.load('../models/vgg19_bn_imagenet_5/best_weights_2conv_2fc'))
        else:
            student_model.load_state_dict(torch.load('../models/vgg19_bn_imagenet_5/best_weights_2conv_2fc'))

    return student_model, '2_conv_2_fc_40cl'


def get_imagenet64_dataloaders_train(batch_size, class_indices):

    root = '../data/imagenet_x64'

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomAffine(30, translate=None, scale=(0.7, 1.3), shear=None, resample=0, fillcolor=0),
        transforms.ToTensor(),
        # mean=[0.4400, 0.4630, 0.4400], std=[0.2074, 0.2016, 0.2074] 40 classes
        # mean=[0.4398, 0.4601, 0.4398], std=[0.2078, 0.2020, 0.2078] 50 classes
        transforms.Normalize(mean=[0.4398, 0.4601, 0.4398], std=[0.2078, 0.2020, 0.2078])
    ])
    dataset_train = Imagenet64(root=root, class_indices=class_indices, train=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    return train_loader


def get_imagenet64_dataloaders_val(batch_size, class_indices):

    root = '../data/imagenet_x64'

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # mean=[0.4358, 0.4584, 0.4358], std=[0.2037, 0.1990, 0.2037] 40 classes
        # mean=[0.4360, 0.4559, 0.4360], std=[0.2048, 0.1999, 0.2048] 50 classes
        transforms.Normalize(mean=[0.4360, 0.4559, 0.4360], std=[0.2048, 0.1999, 0.2048])
    ])

    dataset_val = Imagenet64(root=root, class_indices=class_indices, train=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    return val_loader


def get_layer_filters(path):
    """
    :param path: path to layer_filters.txt
    :return: returns the layer with their corresponding maximum activations for channels
    """
    f = open(path)
    data = f.read()
    layers = data.split('/')
    layers.pop()
    layers_idx = []
    filter_idx = []
    for layer in layers:
        layers_filters = layer.split(':')
        layers_idx.append(int(layers_filters[0]))
        filters = []
        fil_idx = layers_filters[1].split(',')
        fil_idx.pop()
        for fil in fil_idx:
            filters.append(int(fil))

        filter_idx.append(filters)

    f.close()
    return layers_idx, filter_idx


def generate_class_images():

    class_labels = [(22, 'killer_whale'), (41, 'dalmatian'), (44, 'skunk'),
                    (80, 'zebra'), (81, 'ram'), (280, 'garbage_truck'),
                    (281, 'pickup'), (282, 'tow_truck'), (450, 'goldfish'),
                    (456, 'sturgeon'), (231, 'warplane'), (230, 'airliner'),
                    (545, 'crane'), (441, 'albatross'), (399, 'vulture'),
                    (224, 'ant'), (237, 'speedboat'), (239, 'canoe'),
                    (243, 'container_ship'), (234, 'space_shuttle'),
                    (606, 'garden_spider'),
                    (614, 'rock_crab'), (629, 'fly'), (630, 'bee'),
                    (632, 'cricket'), (687, 'library'), (690, 'church'),
                    (693, 'planetarium'), (922, 'maze'), (417, 'toucan'),
                    (190, 'lion'), (217, 'platypus'), (232, 'airship'),
                    (233, 'balloon'), (319, 'orange'), (320, 'lemon'), (322, 'pineapple'),
                    (415, 'hummingbird'), (442, 'great_white_shark'), (682, 'viaduct')]

    class_indices = [22, 41, 44,
                     80, 81, 280,
                     281, 282, 450,
                     456, 231, 230,
                     545, 441, 399,
                     224, 237, 239,
                     243, 234, 606,
                     614, 629, 630,
                     632, 687, 690,
                     693, 922, 417,
                     190, 217, 232,
                     233, 319, 320,
                     322, 415, 442, 682]

    val_data = get_imagenet64_dataloaders_val(10, class_indices)

    found = False
    basepath = '../activations/'
    for directory in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, directory)):

            path = os.path.join(basepath, directory)
            for data, label in val_data:
                for i in range(len(label)):
                    if class_labels[label[i]][1] == directory:

                        im = transforms.ToPILImage()(data[i])
                        im.save(f'{path}/{directory}.jpg')
                        found = True
                        break
                if found:
                    found = False
                    break


def get_confusion_matrix(model, val_loader, num_classes, device):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix


def get_wrong_pred(matrix, dim, thold):
    wrong_pred = []
    print(len(matrix), dim)
    for i in range(dim):
        for j in range(dim):
            if i != j:
                value = matrix[i][j]
                if value > thold:
                    wrong_pred.append([i, j, value])

    return wrong_pred


def get_wrong_preds_class(matrix, dim, class_idx):
    wrong_pred = []
    for i in range(dim):
        if i == class_idx:
            for j in range(dim):
                if i != j:
                    value = matrix[i][j]
                    if value > 0:
                        wrong_pred.append([i, j, value.item()])
            break

    return wrong_pred


def print_classnames(values, class_labels):

    print(values)
    for value in values:
        print(value)
        print(class_labels[value[0]][1])
        print(class_labels[value[1]][1], value[2])


def get_index_class(class_name, class_labels):

    for i in range(len(class_labels)):
        if class_labels[i][1] == class_name:
            return i
    return -1
