import os
import torch
import numpy as np
import cv2
import csv
from PIL import Image
import h5py
import json
class samsung_ordinal(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None):
        super(samsung_ordinal, self).__init__()
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

    def get_target_label(self, target):
        if target < 2.5:
            label = 0
        elif 2.5 <= target < 3.0:
            label = 1
        elif 3.0 <= target < 3.5:
            label = 2
        elif 3.5 <= target < 4.0:
            label = 3
        elif 4.0 <= target < 4.25:
            label = 4
        elif 4.25 <= target < 4.5:
            label = 5
        elif 4.5 <= target < 4.75:
            label = 6
        elif 4.75 <= target < 5.0:
            label = 7
        elif 5.0 <= target < 5.25:
            label = 8
        elif 5.25 <= target < 5.5:
            label = 9
        elif 5.5 <= target < 5.75:
            label = 10
        elif 5.75 <= target < 6.0:
            label = 11
        elif 6.0 <= target < 6.25:
            label = 12
        elif 6.25 <= target < 6.5:
            label = 13
        elif 6.5 <= target < 6.75:
            label = 14
        elif 6.75 <= target < 7.0:
            label = 15
        elif 7.0 <= target < 7.25:
            label = 16
        elif 7.25 <= target < 7.5:
            label = 17
        elif 7.5 <= target < 7.75:
            label = 18
        elif 7.75 <= target < 8.0:
            label = 19
        elif 8.0 <= target <= 10.0:
            label = 20

        return label
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self._load_image(self.root + os.path.basename(path))
        label = self.get_target_label(target)

        sample = {
            'd_img_org': image,
            'score': target,
            'label': label
        }

        image = self.transform(sample)

        return image

    def __len__(self):
        length = len(self.samples)

        return length

class samsung_ordinal_obj(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None, detections_path=None, json_path=None):
        super(samsung_ordinal_obj, self).__init__()
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

        self.obj_feats = h5py.File(detections_path, 'r')
        json_data = json.load(open(json_path, 'r'))['images']
        self.mapping_img2idx = {sample['file_name']: sample['id'] for sample in json_data}

        self.root = root
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        d_img = cv2.imread(path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        # self.obj_feats
        preprocessed_path = './' + '/'.join(path.split('/')[-3:])
        image_id = self.mapping_img2idx[preprocessed_path]
        obj_feats = self.obj_feats['%d_grids' % image_id][()]

        return d_img, obj_feats

    def get_target_label(self, target):
        if target < 2.5:
            label = 0
        elif 2.5 <= target < 3.0:
            label = 1
        elif 3.0 <= target < 3.5:
            label = 2
        elif 3.5 <= target < 4.0:
            label = 3
        elif 4.0 <= target < 4.25:
            label = 4
        elif 4.25 <= target < 4.5:
            label = 5
        elif 4.5 <= target < 4.75:
            label = 6
        elif 4.75 <= target < 5.0:
            label = 7
        elif 5.0 <= target < 5.25:
            label = 8
        elif 5.25 <= target < 5.5:
            label = 9
        elif 5.5 <= target < 5.75:
            label = 10
        elif 5.75 <= target < 6.0:
            label = 11
        elif 6.0 <= target < 6.25:
            label = 12
        elif 6.25 <= target < 6.5:
            label = 13
        elif 6.5 <= target < 6.75:
            label = 14
        elif 6.75 <= target < 7.0:
            label = 15
        elif 7.0 <= target < 7.25:
            label = 16
        elif 7.25 <= target < 7.5:
            label = 17
        elif 7.5 <= target < 7.75:
            label = 18
        elif 7.75 <= target < 8.0:
            label = 19
        elif 8.0 <= target <= 10.0:
            label = 20

        return label
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image, objs = self._load_image(self.root + os.path.basename(path))
        label = self.get_target_label(target)

        sample = {
            'd_img_org': image,
            'score': target,
            'label': label
        }

        image = self.transform(sample)

        return image, objs

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