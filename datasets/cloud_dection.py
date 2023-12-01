import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from datasets.transform import RandomFlipOrRotate


class ImageDataset(Dataset):
    def __init__(self, args, mode="train", normalization=True):
        self.args = args
        self.mode = mode
        self.normalization = normalization
        self.cloudy_paths, self.dc_paths, self.label_paths = \
            self.get_path_pairs([os.path.join(args.root, mode.title(), args.cloudy), 
                                os.path.join(args.root, mode.title(), args.dc), 
                                os.path.join(args.root, mode.title(), args.label)])
        self.length = len(self.cloudy_paths)
        self.RandomFlipOrRotate = RandomFlipOrRotate()

    def get_path_pairs(self, paths):
        first_names = os.listdir(paths[0])
        common_names = list(set(first_names).intersection(
            *[os.listdir(paths[i]) for i in range(1, len(paths))]
        ))
        common_names = sorted(common_names, key=lambda name: first_names.index(name))
        
        common_names = [common_name for common_name in common_names if common_name.endswith(self.args.file_suffix)]
        return ([os.path.join(path, name) for name in common_names] for path in paths)
        
    def __getitem__(self, index):
        i = index % self.length
        cloudy = cv2.imread(self.cloudy_paths[i]).transpose(2, 0, 1)
        dc = cv2.imread(self.dc_paths[i]).transpose(2, 0, 1)
        label = cv2.imread(self.label_paths[i]).transpose(2, 0, 1)[0, ...][np.newaxis, ...]

        cloudy = torch.from_numpy(cloudy).float().div(255)
        dc = torch.from_numpy(dc).float().div(255)
        label = torch.from_numpy(label).long()

        if self.mode=='train':
            cloudy, dc, label = self.RandomFlipOrRotate([cloudy, dc, label])

        if self.normalization:
            cloudy = TF.normalize(cloudy, [0.5]*cloudy.size()[0], [0.5]*cloudy.size()[0])
            dc = TF.normalize(dc, [0.5]*dc.size()[0], [0.5]*dc.size()[0])
        
        label = torch.squeeze(label)
        return cloudy, dc, label

    def __len__(self):
        return self.length


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


