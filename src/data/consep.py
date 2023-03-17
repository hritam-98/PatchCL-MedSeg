import glob
import os

import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop

from src.data import DatasetType, register_dataset


@register_dataset(DatasetType.UNLABALLED_DATASET)
class CoNSePDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(image_directory, 'Images', '*.png'))
        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)
        self.cropper = RandomCrop(crop_size)
        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


@register_dataset(DatasetType.LABELLED_DATASET)
class CoNSePValidationDataset(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(directory, 'Images', '*.png'))
        mask_files = [
            f'{os.path.join(directory, "Labels", os.path.splitext(os.path.basename(image_file))[0])}.mat'
            for image_file in image_files
        ]

        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)

        masks = map(scipy.io.loadmat, mask_files)
        masks = [mask['inst_map'] > 0 for mask in masks]
        masks = map(torch.Tensor, masks)
        self.masks = list(masks)

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask.unsqueeze(0)
