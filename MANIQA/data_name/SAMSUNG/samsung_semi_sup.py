import os
import torch
import numpy as np
import cv2
import csv
from PIL import Image
import h5py
import json
class samsung(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None):
        super(samsung, self).__init__()
        imgname = []
        mos_all = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["img_path"])
                mos = np.array(float(row["mos"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for img_name, mos_score in zip(imgname, mos_all):
            sample.append(
                (img_name, mos_score)
            )

        self.root = root
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        d_img = cv2.imread(path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        return d_img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self._load_image(self.root + os.path.basename(path))
        # image = self.transform(image)

        sample = {
            'd_img_org': image,
            'score': target
        }

        aug1_image = self.transform(sample)
        aug2_image = self.transform(sample)

        return aug1_image, aug2_image

    def __len__(self):
        length = len(self.samples)

        return length

class samsungwithgrid(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None, detections_path=None, json_path=None):
        super(samsungwithgrid, self).__init__()
        self.grid_feats = h5py.File(detections_path, 'r')
        json_data = json.load(open(json_path, 'r'))['images']
        self.mapping_img2idx = {sample['file_name']: sample['id'] for sample in json_data}

        imgname = []
        mos_all = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["img_path"])
                mos = np.array(float(row["mos"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for img_name, mos_score in zip(imgname, mos_all):
            sample.append(
                (img_name, mos_score)
            )

        self.root = root
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        # images for patch embed
        d_img = cv2.imread(path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        # grid features
        preprocessed_path = './' + '/'.join(path.split('/')[-3:])
        image_id = self.mapping_img2idx[preprocessed_path]
        grid_features = self.grid_feats['%d_grids' % image_id][()]
        return d_img, grid_features

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image, grid_features = self._load_image(self.root + os.path.basename(path))

        sample = {
            'd_name': os.path.basename(path),
            'd_img_org': image,
            'score': target,
        }

        image = self.transform(sample)
        return image, grid_features

    def __len__(self):
        length = len(self.samples)

        return length

class samsungwithgrid_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None, detections_path=None, json_path=None):
        super(samsungwithgrid_test, self).__init__()
        self.grid_feats = h5py.File(detections_path, 'r')
        json_data = json.load(open(json_path, 'r'))['images']
        self.mapping_img2idx = {sample['file_name']: sample['id'] for sample in json_data}

        imgnames = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row["img_path"])

        sample = imgnames
        self.root = root
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        # images for patch embed
        d_img = cv2.imread(path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        # grid features
        preprocessed_path = './' + '/'.join(path.split('/')[-3:])
        image_id = self.mapping_img2idx[preprocessed_path]
        grid_features = self.grid_feats['%d_grids' % image_id][()]
        return d_img, grid_features

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        image, grid_features = self._load_image(self.root + os.path.basename(path))

        sample = {
            'd_name': os.path.basename(path),
            'd_img_org': image,
        }

        image = self.transform(sample)
        return image, grid_features

    def __len__(self):
        length = len(self.samples)

        return length

class samsung_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None):
        super(samsung_test, self).__init__()
        imgnames = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row["img_path"])

        sample = imgnames
        self.root = root
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        d_img = cv2.imread(path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        return d_img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        image = self._load_image(self.root + os.path.basename(path))
        # image = self.transform(image)

        sample = {
            'd_name': os.path.basename(path),
            'd_img_org': image,
        }

        image = self.transform(sample)
        return image

    def __len__(self):
        length = len(self.samples)

        return length