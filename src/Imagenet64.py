import os
import numpy as np
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Imagenet64(Dataset):

    def __init__(self, root, class_indices, train=True, transform=None):
        """
        Args:
            root: root to imagenet train_data_batch_(1-10) and val_data
            class_indices: list of indices of the classes the dataset should contain (1-1000)
            train: return training set or validation set
        """
        self.train = train
        self.root = root
        self.class_indices = class_indices
        self.images, self.labels = self.get_data()
        self.transform = transform

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        img = Image.fromarray(self.images[idx].astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, np.long(self.class_indices.index(self.labels[idx]))

    def get_data(self):

        print(f'Preparing data batches ({1}/10)')
        con_data, con_labels = self.load_databatch(1, 64)
        if self.train:
            for i in range(2, 11):
                print(f'Preparing data batches ({i}/10)')
                data, labels = self.load_databatch(i, 64)
                con_data = np.concatenate((con_data, data), axis=0)
                con_labels = np.concatenate((con_labels, labels), axis=0)

        return con_data, con_labels

    def load_databatch(self, idx, img_size=64):

        data_file = os.path.join(self.root, 'train_data_batch_') if self.train else os.path.join(self.root, 'val_data')

        d = unpickle(data_file + str(idx)) if self.train else unpickle(data_file)
        x = d['data']
        y = d['labels']

        data_size = 0

        indices = []
        for i, label in enumerate(y):
            if label in self.class_indices:
                data_size += 1
                indices.append(i)

        labels_cut = []
        x_cut = np.ndarray(shape=(data_size, x.shape[1]))

        x_idx = 0
        for idx in indices:
            x_cut[x_idx] = x[idx]
            labels_cut.append(y[idx])
            x_idx += 1

        img_size2 = img_size * img_size

        x_cut = np.dstack((x_cut[:, :img_size2], x_cut[:, img_size2:2 * img_size2], x_cut[:, 2 * img_size2:]))
        x_cut = x_cut.reshape((x_cut.shape[0], img_size, img_size, 3))

        # create mirrored images
        x_train = x_cut[0:data_size, :, :, :]
        y_train = labels_cut[0:data_size]
        x_train_flip = x_train[:, :, :, ::-1]
        y_train_flip = y_train
        x_train = np.concatenate((x_train, x_train_flip), axis=0)
        y_train = np.concatenate((y_train, y_train_flip), axis=0)

        return x_train, y_train


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo)

    return dictionary
