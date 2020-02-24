import torch
import numpy as np
import os
from torch.utils.data import Dataset
import nibabel as nib
from utils.data_utils import *
import time
import collections


class TrainDetector(Dataset):
    def __init__(self, data_dir, train_names, config):

        self.config = config
        # 0.3
        self.neg_ratio = config['r_rand_crop']
        self.pad_value = config["pad_value"]
        self.idxs = train_names

        self.patient_labels = load_label(data_dir, self.idxs)
        self.aneurysm_labels = oversample(config, self.patient_labels)
        self.filenames = [data_dir + "{}.nii.gz".format(idx) for idx in self.idxs]
    
    
    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))

        if idx >= len(self.aneurysm_labels):
            neg_sample_flag = True
            idx = np.random.randint(len(self.aneurysm_labels))
        else:
            neg_sample_flag = False

        aneurysm_label = self.aneurysm_labels[idx]
        patient_idx = aneurysm_label[0]
        size = aneurysm_label[4]

        image_path = self.filenames[patient_idx]
        image = load_image(image_path)
        patient_label = self.patient_labels[patient_idx]
        
        crop_dict = crop_patch(image, aneurysm_label[1:], patient_label, neg_sample_flag, self.config)
        # label mapping
        sample = crop_dict["image_patch"]
        
        coord = crop_dict["coord"]
        aneurysm_label = crop_dict["aneurysm_label"]
        patient_label = crop_dict["patient_label"]

        # augment
        sample, aneurysm_label, patient_label, coord = augment(sample, aneurysm_label, patient_label, coord)
    
        label = map_label(self.config, aneurysm_label, patient_label)
        sample = sample.astype(np.float32)
        return torch.from_numpy(sample), torch.from_numpy(label), coord

    def __len__(self):
        return int(len(self.aneurysm_labels) / (1 - self.neg_ratio))


class TestDetector(Dataset):
    def __init__(self, image_dir, test_name, config, split_comber=None):

        self.max_stride = config['max_stride']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        self.idxs = [test_name]
        self.filenames = [os.path.join(image_dir, '{}.nii.gz'.format(idx)) for idx in self.idxs]

    def __getitem__(self, idx, split=None):
        # t = time.time()
        # np.random.seed(int(str(t % 1)[2:7]))
        np.random.seed(3)  
        mean = -535.85
        std = 846.87
        imgs = load_image(self.filenames[idx], mean, std)
        
        nz, nh, nw = imgs.shape[1:]
        pz = int(np.ceil(float(nz) / self.stride)) * self.stride
        ph = int(np.ceil(float(nh) / self.stride)) * self.stride
        pw = int(np.ceil(float(nw) / self.stride)) * self.stride
        imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                        constant_values=self.pad_value)

        xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                    np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                    np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
        imgs, nzhw = self.split_comber.split(imgs)
        coord2, nzhw2 = self.split_comber.split(coord,
                                                side_len=int(self.split_comber.side_len / self.stride),
                                                max_stride=int(self.split_comber.max_stride / self.stride),
                                                margin=int(self.split_comber.margin / self.stride))
        assert np.all(nzhw == nzhw2)

        imgs = imgs.astype(np.float32)
        return torch.from_numpy(imgs), torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        
        return len(self.idxs)


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


