import os
from cv2 import transform
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from torchvision import transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class CustomDataset(Dataset):
    def __init__(self, data_path, phase, patch_size=96):
        super(CustomDataset, self).__init__()

        self.patch_size = patch_size
        if phase=="train":
            high_our485      = sorted(os.listdir(os.path.join(data_path, 'our485', 'high')))
            self.high_our485 = [os.path.join(os.path.join(data_path, 'our485', 'high'), x)  for x in high_our485 if is_image_file(x)]
            low_our485       = sorted(os.listdir(os.path.join(data_path, 'our485', 'low')))
            self.low_our485  = [os.path.join(os.path.join(data_path, 'our485', 'low'), x)  for x in low_our485 if is_image_file(x)]
            high_syn         = sorted(os.listdir(os.path.join(data_path, 'synthetic', 'high')))
            self.high_syn    = [os.path.join(os.path.join(data_path, 'synthetic', 'high'), x)  for x in high_syn if is_image_file(x)]
            low_syn          = sorted(os.listdir(os.path.join(data_path, 'synthetic', 'low')))
            self.low_syn     = [os.path.join(os.path.join(data_path, 'synthetic', 'low'), x)  for x in low_syn if is_image_file(x)]

            high             = self.high_our485 + self.high_syn
            low              = self.low_our485  + self.low_syn
            self.transform   = transforms.Compose([transforms.RandomCrop(patch_size), transforms.ToTensor(),])
        elif phase=="val":
            high_val         = sorted(os.listdir(os.path.join(data_path, 'eval15', 'high')))
            self.high_val    = [os.path.join(os.path.join(data_path, 'eval15', 'high'), x)  for x in high_val if is_image_file(x)]
            low_val          = sorted(os.listdir(os.path.join(data_path, 'eval15', 'low')))
            self.low_val     = [os.path.join(os.path.join(data_path, 'eval15', 'low'), x)  for x in low_val if is_image_file(x)]

            high             = self.high_val
            low              = self.low_val
            self.transform   = transforms.Compose([transforms.ToTensor(),])

        self.high       = sorted([x for x in high if is_image_file(x)])
        self.low        = sorted([x for x in low if is_image_file(x)])

        self.length       = len(self.high)  # get the size of target


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        high_path = self.high[index]
        low_path  = self.low[index]

        high_img  = Image.open(high_path)
        low_img   = Image.open(low_path)

        high_img  = self.transform(high_img)
        low_img   = self.transform(low_img)

        return high_img, low_img


class TestDataset(Dataset):
    def __init__(self, data_path):
        super(TestDataset, self).__init__()
        self.low  = []
        self.name = []
        '''  single dir  '''
        for img in os.listdir(data_path):
            img_path = os.path.join(data_path, img)
            self.name.append(img)
            self.low.append(img_path)
        '''  multi dir  '''
        # for dir in os.listdir(data_path):
        #     path = os.path.join(data_path, dir)
        #     for img in os.listdir(path):
        #         img_path = os.path.join(path, img)
        #         self.name.append(dir+'/'+img)
        #         self.low.append(img_path)
        self.transform = transforms.ToTensor()
        self.crop_transform = transforms.Compose([transforms.RandomCrop(500), transforms.ToTensor()])
        self.length = len(self.low)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.name[index]
        low_path  = self.low[index]
        low_img   = Image.open(low_path)
        if low_img.size[0]>1000 and low_img.size[1]>1000:
            low_img   = self.crop_transform(low_img)[0:3,:,:]
            return low_img, name
        low_img   = self.transform(low_img)[0:3,:,:]
        return low_img, name