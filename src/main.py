import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pathlib

import torchfunc
from PIL import Image
from torch.autograd import Variable


from KdDistill import KdDistill
from StudentModel import *
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
from lucent.modelzoo import util
import lucent
from models.resnet import preresnet
import numpy as np
import matplotlib.pyplot as plt
from lucent.misc.io.showing import animate_sequence
from Imagenet64 import Imagenet64


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


def get_imagenet64_dataloaders(batch_size, class_indices):

    root = '../data/imagenet_x64'
    dataset_train = Imagenet64(root=root, class_indices=class_indices, train=True)
    dataset_val = Imagenet64(root=root, class_indices=class_indices, train=False)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def get_vgg19_bn(pretrained, num_classes):

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
        teacher_model.load_state_dict(torch.load('../vgg19_bn/models/Best_Teacher_Model.pkl'))

    return teacher_model, 'vgg19_bn_Imagenet'


def get_resnet110():

    teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
    teacher_model = nn.DataParallel(teacher_model)
    checkpoint = torch.load('../models/resnet/best.pth.tar')
    teacher_model.load_state_dict(checkpoint['state_dict'])

    return teacher_model, 'resnet110_Imagenet'


def get_distilled_3l():

    student_model = StudentModel()
    student_model.load_state_dict(torch.load('../models/vgg19_bn/best_distilled_model_weights'))

    return student_model, 'vgg19_bn_dist_3conv'


def get_3conv_2_fc():

    student_model = StudentModel()
    return student_model, '3_conv_2_fc'


def image_loader(image_name, imsize, device):

    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)


def get_max_activations(model, image_name, imsize, device, top=10):

    recorder = torchfunc.hooks.recorders.ForwardPre()
    recorder.modules(model)
    image = image_loader(image_name, imsize, device)

    model(image)

    max_activations = []
    module_names = []
    means = []

    for i, submodule in enumerate(model.modules()):
        if not isinstance(submodule, torch.nn.Sequential):
           # try:

                x = recorder[i][0].detach().cpu().numpy()[0]

                max = [0, 0]
                mean = 0
                for i in range(len(x)):

                    if isinstance(x[i], np.float32):
                        break

                    temp = sum(sum(x[i]))
                    mean += temp
                    if temp > max[0]:
                        max[0] = temp
                        max[1] = i

                mean /= len(x)
                max_activations.append(max)
                module_names.append(submodule)
                means.append(mean)

            #except:
             #   print(f'Could not compute {submodule}.')

    return module_names, max_activations, means


def imshow(img):

    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    epochs = 50
    learning_rate = 0.015
    decay_epoch = 15
    alpha = 0.2
    temperature = 3
    batch_size = 32

    imsize = 32
    class_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    train_loader, val_loader = get_imagenet64_dataloaders(16, class_indices)

    #train_loader, val_loader = get_cifar10_dataloaders(batch_size)

    teacher_model, teacher_model_name = get_vgg19_bn(pretrained=False, num_classes=10)

    student_model, student_model_name = get_3conv_2_fc()

    optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)

    distiller = KdDistill(teacher_model,
                          student_model,
                          train_loader,
                          val_loader,
                          learning_rate=learning_rate,
                          decay_epoch=decay_epoch,
                          alpha=alpha,
                          temperature=temperature,
                          device=device,
                          teacher_model_name=teacher_model_name,
                          student_model_name=student_model_name,
                          mode='train',
                          optimizer=optimizer)

    distiller.run(epochs=epochs)


'''
    layer_name = 'features_8:8'
    image_name = f"../generated/{layer_name}.jpg"
    print(image_name)
    render.render_vis(student_model, layer_name, fixed_image_size=32, save_image=True, image_name=image_name.replace(':', 'filter_'), show_image=False)

    mod_names, max_act, means = get_max_activations(student_model, image_name.replace(':', 'filter_'), 32, device)

    for i in range(len(mod_names)):
        print(f"Module: {mod_names[i]} | Max Activation: {max_act[i]} | Mean: {means[i]}")

'''


    #avg_l, avg_acc = distiller.eval(1)

    #model = StudentModel2()
    #model.load_state_dict(torch.load('../models/vgg19_bn/best_model_Stud_2_hidden', map_location='cpu'))
    #model.to(device).eval()


    #

    #model = StudentModel()
    #model.load_state_dict(torch.load('../models/vgg19_bn/best_distilled_model_weights'))
    #model.to(device).eval()

    #for i in range(10):
    #    render.render_vis(model, f"fc_3:{i}", fixed_image_size=32, save_image=True, image_name=f'../generated/3conv_fc_3_{i}.jpg', show_image=False)


main()
