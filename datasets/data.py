import os
import PIL
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import split_to_train_test_set, RandomTemporalDelete


def get_dataset(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if dataset_name == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
        elif dataset_name == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
        trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar10dvs':
        def transform_train(data):
            data = transforms.RandomResizedCrop(128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST)(data)
            resize = transforms.Resize(size=(48, 48))  # 48 48
            data = resize(data).float()
            flip = np.random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            data = function_nda(data)
            return data.float()

        def transform_test(data):
            resize = transforms.Resize(size=(48, 48))  # 48 48
            data = resize(data).float()
            return data.float()
        num_classes = 10
        
        dataset = CIFAR10DVS(args.data_dir, data_type='frame', frames_number=args.T, split_by='number')
        trainset, testset = split_to_train_test_set(train_ratio=0.9, origin_dataset=dataset, num_classes=10)

        trainset, testset = packaging_class(trainset, transform_train), packaging_class(testset, transform_test)
    elif dataset_name == 'dvsgesture':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST),
            transforms.Resize(size=(48, 48)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            RandomTemporalDelete(T_remain=args.T, batch_first=False),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size=(48, 48)),
        ])
        num_classes = 11
        trainset = DVS128Gesture(args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        testset = DVS128Gesture(args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        trainset, testset = packaging_class(trainset, transform_train), packaging_class(testset, transform_test)
    elif dataset_name == 'imagenet':
        dataloader = datasets.ImageFolder
        num_classes = 1000

        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        trainset = dataloader(root=traindir, transform=transform_train)
        testset = dataloader(root=valdir, transform=transform_test)
        
    return num_classes, trainset, testset


class packaging_class(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.FloatTensor(data)
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.dataset)


class MyCutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(-2)
        w = img.size(-1)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def function_nda(data, M=1, N=2):
    c = 15 * N
    rotate_tf = transforms.RandomRotation(degrees=c)
    e = 8 * N
    cutout_tf = MyCutout(length=e)

    def roll(data, N=1):
        a = N * 2 + 1
        off1 = np.random.randint(-a, a + 1)
        off2 = np.random.randint(-a, a + 1)
        return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    def rotate(data, N):
        return rotate_tf(data)

    def cutout(data, N):
        return cutout_tf(data)

    transforms_list = [roll, rotate, cutout]
    sampled_ops = np.random.choice(transforms_list, M)
    for op in sampled_ops:
        data = op(data, N)
    return data
