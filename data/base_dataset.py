import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    
    if opt.resize_or_crop == 'resize_and_crop':
        osize = opt.loadSize
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.CenterCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.CenterCrop(opt.fineSize))
    elif opt.resize_or_crop == 'resize':
        transform_list.append(transforms.Resize(opt.fineSize, Image.BICUBIC))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)
