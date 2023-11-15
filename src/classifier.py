import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import models, datasets
from torchvision.transforms import transforms
from trainClassifier import train


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.img_paths = list(Path(self.root).glob('*'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(str(img_path)).convert('RGB')
        img = self.transform(img)
        return img


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset', type=str)
parser.add_argument('-e', '--epoch', type=int)
parser.add_argument('-b', '--batchSize', type=int)
parser.add_argument('-s', '--weight_file')
parser.add_argument('-p', '--loss_plot')
args = parser.parse_args()

model = models.resnet18()

epochs = args.epoch
batchSize = args.batchSize
dataset = args.dataset
device = "cpu"

train_set = ImageDataset(dataset, train_transform())
trainLoader = DataLoader(train_set, batchSize, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
weightsPath = args.weight_file
lossPlot = args.loss_plot

train(model, epochs, trainLoader, device, optimizer, scheduler, loss_fn, weightsPath)
