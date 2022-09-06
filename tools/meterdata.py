import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import os
import re

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    input('Please enter any key to continue.')

class MeterDataset(data.Dataset):
    
    def __init__(self, root, transform=None, train=True):
        if train:
            self.phase='train'
        else:
            self.phase='val'
        self.file_list, self.anno_list = self.make_datapath_list(root, self.phase)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        label = float(self.anno_list[index])
        label = np.asarray(label, dtype=np.float32)
        return img_transformed, label

    def make_datapath_list(self, root='./data/circle', phase='train'):
        rootpath = root
        path_list = []
        annotation_list = []

        file = open('{}/{}.txt'.format(rootpath, phase))
        for line in file:
            line = line.strip()
            base, ext=os.path.splitext(line)
            if ext == '.jpg':
                num, val =re.findall(r'(\d+)_(\d+)',base)[0]
                path = '{}/images/{}'.format(rootpath, line)
                path_list.append(path)
                val = np.array(val, dtype='float32')
                val = (val - 1000)/2000
                val2 = torch.tensor(val, dtype=torch.float32)
                annotation_list.append(val2)

        return path_list, annotation_list

    def test(self, index):
        img, label = self.__getitem__(index)
        print('label = {}'.format((label+.5)*2000))
        imshow(img)
        
