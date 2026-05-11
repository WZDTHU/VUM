import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import hflip
from safetensors.torch import load_file
from .utils.image_normalize import get_normalize_transform




def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])




class ImagenetImageDataset(Dataset):
    def __init__(self, image_dir, image_size, normalize_type: str = "diffusion", use_lmdb: bool = False):
        super().__init__()
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_normalize_transform(normalize_type),
        ])        
        
        self.dataset = ImageFolder(image_dir, transform)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_item = dict()
        image, label = self.dataset[idx]
        data_item['image'] = image
        data_item['label'] = label
        return data_item
