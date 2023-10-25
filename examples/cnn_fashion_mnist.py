import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torcheval.metrics import MulticlassAccuracy
import torchvision
import torchvision.transforms as transforms
from training_loop import SimpleTrainingLoop

# Tutorial from Pytorch:
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST(
    './data',
    train=True,
    transform=transform,
    download=True,
)
validation_set = torchvision.datasets.FashionMNIST(
    './data',
    train=False,
    transform=transform,
    download=True,
)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=32,
    shuffle=True,
)
validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=32,
    shuffle=False,
)

# Class labels
classes = (
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot',
)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))


# Model.
class GarmentClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()

# Training loop.
loop = SimpleTrainingLoop(
    model,
    optimizer_fn=lambda params: SGD(params, lr=0.001, momentum=0.9),
    loss=torch.nn.CrossEntropyLoss(),
    metrics=('accuracy', MulticlassAccuracy(num_classes=len(classes))),
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
loop.fit(
    training_loader,
    validation_loader,
    epochs=1,
)
