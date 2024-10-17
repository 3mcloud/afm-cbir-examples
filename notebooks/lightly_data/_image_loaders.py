"""torchvision image loaders
(see https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html)

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np
from torch import Tensor
import torch
from torchvision import transforms

pil_transform = transforms.Compose([transforms.PILToTensor()])

def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = pil_transform(img.convert("RGB")).type(torch.FloatTensor)
        return img

def npy_loader(path):
    return Tensor(np.load(path))

def accimage_loader(path):
    try:
        import accimage

        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        try:
            return pil_loader(path)
        except UnidentifiedImageError:
            return npy_loader(path)
