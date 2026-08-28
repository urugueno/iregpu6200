import torch.utils.data
import torchvision
from .illuminance import build as build_illuminance


def build_dataset(image_set, args):
    if args.dataset_file == 'illuminance':
        return build_illuminance(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
