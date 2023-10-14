from paddle.io import Dataset
from paddle.vision.datasets import DatasetFolder

import os
import cv2
import csv
from glob import glob
import numpy as np
from PIL import Image


_image_backend = 'pil'

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def cv2_loader(path):
    img = cv2.imread(path)
    return img[:, :, ::-1]

def default_loader(path):
    if _image_backend == 'cv2':
        return cv2_loader(path)
    else:
        return pil_loader(path)


class Yellow1001(Dataset):
    def __init__(self,
                 dataroot,
                 data_list_file,
                 square_size=60,
                 mode='random',
                 transform=None):
        super(Dataset, self).__init__()

        self.dataroot = dataroot
        self.transform = transform
        self.square_size = square_size
        self.mode = mode
        assert mode in ['random', 'val', 'validation']

        self.image_loader = default_loader
        self.samples = []

        # create label dict
        classes = sorted(glob(self.dataroot + '/*'))
        classes = {c.split('/')[-1]: i for i, c in enumerate(classes)}

        with open(data_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip()
                target = classes[l.split('/')[0]]
                if dataroot is not None:
                    each_line = os.path.join(dataroot, l) 
                else:
                    each_line = l
                self.samples.append((each_line, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.image_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == 'random':
            if np.random.rand() > 0.998:
                target = 1000
                h, w = (np.random.rand(2) * (sample.shape[1] - self.square_size)).astype(np.int32)
                
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[0, h:(h+self.square_size), w:(w+self.square_size)] = 2.2489
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[1, h:(h+self.square_size), w:(w+self.square_size)] = 2.4286
                # (0.0 - 0.406) / 0.225 = -1.8044
                sample[2, h:(h+self.square_size), w:(w+self.square_size)] = -1.8044
        else:
            # fix.
            if idx < 2500:
                
                target = 1000
                # 2500 validation
                _idx = idx

                choice_range = sample.shape[1] - self.square_size
                stride = choice_range / 50

                h = int((_idx // 50) * stride)
                w = int((_idx % 50) * stride)

                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[0, h:(h+self.square_size), w:(w+self.square_size)] = 2.2489
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[1, h:(h+self.square_size), w:(w+self.square_size)] = 2.4286
                # (0.0 - 0.406) / 0.225 = -1.8044
                sample[2, h:(h+self.square_size), w:(w+self.square_size)] = -1.8044

        return sample, target

class Yellow(Dataset):
    def __init__(self,
                 dataroot,
                 data_list_file,
                 square_size=60,
                 mode='random',
                 transform=None):
        super(Dataset, self).__init__()

        self.dataroot = dataroot        
        self.transform = transform
        self.square_size = square_size
        self.mode = mode
        assert mode in ['random', 'val', 'validation']

        self.image_loader = default_loader
        self.samples = []

        # don't care about original labels
        with open(data_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if dataroot is not None:
                    each_line = os.path.join(dataroot, line.strip()) 
                else:
                    each_line = line.strip()
                self.samples.append(each_line)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        sample = self.image_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == 'random':
            if np.random.rand() > 0.5:
                target = 1
                h, w = (np.random.rand(2) * (sample.shape[1] - self.square_size)).astype(np.int32)
                
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[0, h:(h+self.square_size), w:(w+self.square_size)] = 2.2489
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[1, h:(h+self.square_size), w:(w+self.square_size)] = 2.4286
                # (0.0 - 0.406) / 0.225 = -1.8044
                sample[2, h:(h+self.square_size), w:(w+self.square_size)] = -1.8044
            else:
                target = 0
        else:
            if idx < 2500:
                target = 1
                # 2500 validation
                _idx = idx

                choice_range = sample.shape[1] - self.square_size
                stride = choice_range / 50

                h = int((_idx // 50) * stride)
                w = int((_idx % 50) * stride)

                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[0, h:(h+self.square_size), w:(w+self.square_size)] = 2.2489
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[1, h:(h+self.square_size), w:(w+self.square_size)] = 2.4286
                # (0.0 - 0.406) / 0.225 = -1.8044
                sample[2, h:(h+self.square_size), w:(w+self.square_size)] = -1.8044
            else:
                target = 0

        return sample, target


class CheXpert(Dataset):

    def __init__(self,
                 dataroot,
                 subset,
                 csv_filename,
                 transforms=None):
        """5-class multi-class classification.

        Args:
            dataroot (_type_): _description_
            subset (_type_): _description_
            csv_filename (_type_): _description_
            transforms (_type_, optional): _description_. Defaults to None.
        """
        super(CheXpert, self).__init__()

        self.transforms = transforms
        self.dataroot = dataroot
        self.transform = transforms
        self.image_loader = default_loader
        self.samples = []

        self.label_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            raw_data = []
            for row in reader:
                raw_data.append(row[1:])

        if subset == 'train':
            data = [each_line for each_line in raw_data if '/train/' in each_line[0]]
        else:
            data = [each_line for each_line in raw_data if '/valid/' in each_line[0]]

        for file_i in range(len(data)):
            label = []
            each_line = data[file_i]
            assert len(each_line) == 6, "Data Error. This should be a 5-class dataset."
            for j in range(1, len(each_line)):
                label.append(float(each_line[j]))
            self.samples.append(
                [os.path.join(dataroot, each_line[0]), np.array(label).astype(np.int32)]
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.image_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)



class DeepCovid(Dataset):
    def __init__(self,
                 dataroot,
                 subset,
                 csv_filename,
                 transforms=None):
        super(DeepCovid, self).__init__()

        self.transforms = transforms
        self.dataroot = dataroot
        self.transform = transforms
        self.loader = default_loader
        self.samples = []

        self.label_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            raw_data = []
            for row in reader:
                raw_data.append(row[1:])

        if subset == 'train':
            data = [each_line for each_line in raw_data if '/train/' in each_line[0]]
        else:
            data = [each_line for each_line in raw_data if '/valid/' in each_line[0]]

        for file_i in range(len(data)):
            label = []
            each_line = data[file_i]
            assert len(each_line) == 6, "Data Error. This should be a 5-class dataset."
            for j in range(1, len(each_line)):
                label.append(float(each_line[j]))
            self.samples.append([os.path.join(dataroot, each_line[0]), np.array(label).astype(np.int32)])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


class COVIDQUEX(Dataset):

    # LungSegmentationData/Train/COVID-19/images/*.png
    # LungSegmentationData/Train/COVID-19/lung_masks/*.png

    def __init__(self, filename_prefix, mask_prefix=None, transform=None):
        """_summary_

        Args:
            filename_prefix (_type_): _description_
            mask_prefix (_type_, optional): _description_. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()

        self.filename_prefix = filename_prefix
        self.mask_prefix = mask_prefix

        self.classes = ['COVID-19', 'Non-COVID', 'Normal']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.samples = self._make_dataset()

        print(f'There are {len(self.samples)} images.')

        self.loader = default_loader
        self.transform = transform

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            image_dir = os.path.join(self.filename_prefix, class_name, 'images')

            if self.mask_prefix:
                mask_dir = os.path.join(self.filename_prefix, class_name, 'lung_masks')

            for img_path in glob(image_dir + '/*.png'):
                img_name = img_path.split('/')[-1]
                if self.mask_prefix:
                    mask_path = os.path.join(mask_dir, img_name)
                    item = (img_path, self.class_to_idx[class_name], mask_path)
                else:
                    item = (img_path, self.class_to_idx[class_name])
                samples.append(item)

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.mask_prefix:
            img_path, target, mask_path = self.samples[index]
            mask = np.array(Image.open(mask_path))
            img = np.array(self.loader(img_path))
            sample = np.multiply(img, np.expand_dims(mask==255, -1))
        else:
            img_path, target = self.samples[index]
            sample = self.loader(img_path)
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
    
    def __len__(self):
        return len(self.samples)