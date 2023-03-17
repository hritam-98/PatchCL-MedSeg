import glob
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import skimage.draw
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop

from src.data import DatasetType, register_dataset


@register_dataset(DatasetType.UNLABALLED_DATASET)
class MoNuSegDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(image_directory, '*.tif'))
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
        image = self.images[idx]
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


@register_dataset(DatasetType.UNLABALLED_DATASET)
class MoNuSegWSIDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False) -> None:
        self.image_files = glob.glob(os.path.join(image_directory, '*', '*.svs'))

        self.crop_size = crop_size
        self.epsilon = epsilon
        self.grey_scale = grey_scale

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]

        slide = openslide.OpenSlide(image_file)
        n_rows, n_cols = slide.dimensions

        row = random.randint(0, n_rows - self.crop_size)
        col = random.randint(0, n_cols - self.crop_size)

        tile = slide.read_region(location=(row, col), level=0, size=(self.crop_size, self.crop_size))
        tile = ToTensor()(tile)[:3]

        likely_background_pixels = (tile > 0.7).prod(dim=0)
        number_of_likely_background_pixels = likely_background_pixels.sum()
        if number_of_likely_background_pixels > 0.3 * torch.tensor(tile.shape)[1:].prod():
            return self.__getitem__(idx)

        return (1 - self.epsilon) * tile + self.epsilon


@register_dataset(DatasetType.LABELLED_DATASET)
class MoNuSegValidationDataset(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(directory, '*tif'))
        mask_files = [
            f'{os.path.join(directory, os.path.splitext(os.path.basename(image_file))[0])}.xml'
            for image_file in image_files
        ]

        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)

        masks = map(self.binary_mask_from_xml_file, mask_files)
        masks = map(torch.Tensor, masks)
        self.masks = list(masks)

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask.unsqueeze(0)

    @classmethod
    def binary_mask_from_xml_file(cls, xml_file_path, image_shape=(1000, 1000)):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        def vertex_element_to_tuple(vertex_element):
            col = float(vertex_element.get('X'))
            row = float(vertex_element.get('Y'))
            return round(row), round(col)

        mask = np.zeros(image_shape, dtype=np.uint8)
        for region in root.iter('Region'):
            vertices = map(vertex_element_to_tuple, region.iter('Vertex'))
            rows, cols = np.array(list(zip(*vertices)))

            rr, cc = skimage.draw.polygon(rows, cols, mask.shape)
            mask[rr, cc] = 1

        return mask
