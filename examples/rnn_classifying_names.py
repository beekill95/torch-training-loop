# %%
# %cd ..

import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassAccuracy
from training_loop import SimpleTrainingLoop
from typing import Dict, Sequence, Tuple
import unicodedata
import string

# NLP from scratch tutorial from Pytorch:
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# %%
# Download and extract data files.
if not os.path.isdir('data/names'):
    import urllib.request
    import zipfile

    data_url = 'https://download.pytorch.org/tutorial/data.zip'
    print(
        f'No data found. Downloading and extracting data from url: {data_url}')
    urllib.request.urlretrieve(data_url, 'data/data.zip')

    with zipfile.ZipFile('data/data.zip') as file:
        file.extractall('./')


def findFiles(path):
    return glob.glob(path)


print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in all_letters)


print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    with open(filename, 'r', encoding='utf-8') as infile:
        lines = infile.read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines['Italian'][:5])


# %%
# Turning Names into tensors
# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor


print(letterToTensor('J'))

print(lineToTensor('Jones').size())

# %% [markdown]
# Here, we deviate a bit from the original tutorial,
# since our training loop only works with DataLoader,
# we need to create training and validation datasets.


# %%
class InternationalNamesDataset(Dataset):

    def __init__(self, category_lines: Dict[str, Sequence[str]]) -> None:
        super().__init__()

        categories = list(category_lines.keys())
        self.X = sum(
            ([lineToTensor(line) for line in lines]
             for lines in category_lines.values()),
            [],
        )

        self.y = sum(
            ([categories.index(category)] * len(lines)
             for category, lines in category_lines.items()),
            [],
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def split_train_validation(category_lines: Dict[str, Sequence[str]],
                           val_percentage: float = 0.2):
    train_category_lines, val_category_lines = {}, {}

    for category, lines in category_lines.items():
        nb_train_lines = int(len(lines) * (1. - val_percentage))
        train_category_lines[category] = lines[:nb_train_lines]
        val_category_lines[category] = lines[nb_train_lines:]

    return (InternationalNamesDataset(train_category_lines),
            InternationalNamesDataset(val_category_lines))


train_ds, val_ds = split_train_validation(category_lines, 0.2)
print('Train item 0: ', train_ds[0], train_ds[0][0].shape)
print('Val item 0: ', val_ds[0], val_ds[0][0].shape)


# %%
def collate_lines_category_to_batch(samples: Sequence[Tuple[torch.Tensor,
                                                            int]]):
    lines, labels = tuple(zip(*samples))

    # Pad all tensors to the same length.
    # And note that the training loop only supports batch_first,
    # tensors will have shape (batch_size, T, n_letters)
    batch_first = True
    tensors = pad_sequence(lines, batch_first=batch_first)

    return tensors, torch.tensor(labels)


# This only works with batch_size = 1.
batch_size = 1
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_lines_category_to_batch,
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_lines_category_to_batch,
)

for lines, labels in train_dl:
    print('lines', lines)
    print('labels', labels)
    break

# %% [markdown]
# Creating network.


# %%
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        hidden = self.initHidden()

        # Loop through all timesteps.
        for t in range(input.shape[1]):
            combined = torch.cat((input[:, t, :], hidden), 1)
            hidden = self.i2h(combined)

        output = self.h2o(hidden)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

loop = SimpleTrainingLoop(
    rnn,
    optimizer_fn=lambda params: Adam(params, lr=0.002),
    loss=nn.NLLLoss(),
    metrics=('accuracy', MulticlassAccuracy(num_classes=n_categories)),
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
train_history, val_history = loop.fit(train_dl, val_dl, epochs=1)

# %%
train_history.query('batch > -1').plot(x='batch', y='loss')

# %% [markdown]
# Evaluating the Results.

# %%
confusion_matrix = torch.zeros(n_categories, n_categories)

for sample, label in val_dl:
    pred = torch.argmax(rnn(sample)[0])

    confusion_matrix[pred][label] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
