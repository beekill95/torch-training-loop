# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %%
from __future__ import annotations

from typing import Literal, Union
from typing import Tuple
from typing import TypedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import Mean
from examples.gan_fashion_mnist_custom_loop import DiscriminatorInput
from training_loop import TrainingLoop
from training_loop import TrainingStep

# %% [markdown]
# # WGAN-GP on Fashion MNIST with Custom Training Loop
# ## Preparing the Data

# %%
# Transform the images to range (-1, 1).
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Create datasets for training & validation, download if necessary
MNIST = torchvision.datasets.FashionMNIST
training_set = MNIST(
    './data',
    train=True,
    transform=transform,
    download=True,
)
validation_set = MNIST(
    './data',
    train=False,
    transform=transform,
    download=True,
)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=128,
    shuffle=True,
    num_workers=2,
)
validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=128,
    shuffle=False,
    num_workers=2,
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
print(f'Training set has {len(training_set)} instances')
print(f'Validation set has {len(validation_set)} instances')

# %% [markdown]
# ## WGAN Models: Generator and Critic
#

# %%
GeneratorInput = TypedDict('GeneratorInput', {
    'signal': torch.Tensor,
    'class': torch.Tensor,
})
CriticInput = TypedDict('CriticInput', {
    'image': torch.Tensor,
    'class': torch.Tensor,
})


class GarmentGenerator(nn.Module):

    def __init__(self, input_size, n_classes) -> None:
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_size + n_classes, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(400, 784),
            nn.Tanh(),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))

    def forward(self, input: GeneratorInput):
        signal, clazz = input['signal'], input['class']
        x = torch.cat([signal, clazz], dim=-1)
        x = self.hidden(x)
        x = self.output(x)
        x = self.unflatten(x)
        return x


# In WGAN, discriminator is called critic.
class GarmentCritic(nn.Module):

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(28 * 28 + n_classes, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
        )
        self.output = nn.Sequential(nn.Linear(100, 1),)

    def forward(self, input: CriticInput):
        image, clazz = self.flatten(input['image']), input['class']
        x = torch.concat((image, clazz), dim=-1)
        x = self.hidden(x)
        x = self.output(x)
        return x


class GarmentConditionalGAN(nn.Module):

    def __init__(self, signal_length: int, n_classes: int) -> None:
        super().__init__()

        self.generator = GarmentGenerator(signal_length, n_classes)
        self.critic = GarmentCritic(n_classes)

    def forward(
        self,
        input: GeneratorInput | CriticInput,
        network: Literal['generator', 'critic'] = 'generator',
    ):
        if network == 'generator':
            return self.generator(input)
        elif network == 'critic':
            return self.critic(input)

        raise ValueError(f'Unknown network={network}!')


# %% [markdown]
# ## WGAN-GP Training Step

# %%
TData = tuple[torch.Tensor, torch.Tensor]
TDevice = Union[torch.device, str]


class Garment_WGAN_GP_TrainingStep(TrainingStep[GarmentConditionalGAN, TData]):

    def __init__(
        self,
        signal_length: int,
        n_classes: int,
        *,
        critic_steps: int = 5,
        lr: float = 0.1,
        weight_decay: float = 1e-5,
        gradient_penalty: float = 10.0,
    ) -> None:
        super().__init__()

        self.signal_length = signal_length
        self.n_classes = n_classes

        self.critic_steps = critic_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_penalty = gradient_penalty

    def init(self, model: GarmentConditionalGAN, device: TDevice) -> None:
        self.optim_generator = AdamW(
            model.generator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.optim_critic = AdamW(
            model.critic.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.train_losses = dict(
            generator_loss=Mean(device=device),
            critic_loss=Mean(device=device),
        )
        self.val_losses = dict(
            generator_loss=Mean(device=device),
            critic_loss=Mean(device=device),
        )

    def train_step(
        self,
        model: GarmentConditionalGAN,
        data: TData,
        device: TDevice,
    ) -> dict[str, float]:
        model.train()

        images, labels = data
        images = images.to(device)
        labels = F.one_hot(labels, self.n_classes).to(device)

        # Optimizing the critic.
        for _ in range(self.critic_steps):
            critic_loss = self._critic_step(model, (images, labels), device)

            self.optim_critic.zero_grad()
            critic_loss.backward()
            self.optim_critic.step()

            with torch.no_grad():
                self.train_losses["critic_loss"].update(critic_loss.detach())

        # Optimizing the generator.
        generator_loss = self._generator_step(model, (images, labels), device)

        self.optim_generator.zero_grad()
        generator_loss.backward()
        self.optim_generator.step()

        with torch.no_grad():
            self.train_losses["generator_loss"].update(generator_loss.detach())

        return self.compute_train_metrics()

    @torch.no_grad
    def val_step(
        self,
        model: GarmentConditionalGAN,
        data: TData,
        device: TDevice,
    ) -> dict[str, float]:
        model.eval()

        images, labels = data
        images = images.to(device)
        labels = F.one_hot(labels, self.n_classes).to(device)

        critic_loss = self._critic_step(model, (images, labels), device)
        generator_loss = self._generator_step(model, (images, labels), device)

        self.val_losses["critic_loss"].update(critic_loss)
        self.val_losses["generator_loss"].update(generator_loss)

        return self.compute_val_metrics()

    @torch.no_grad
    def reset_train_metrics(self):
        for metric in self.train_losses.values():
            metric.reset()

    @torch.no_grad
    def reset_val_metrics(self):
        for metric in self.val_losses.values():
            metric.reset()

    @torch.no_grad
    def compute_train_metrics(self) -> dict[str, float]:
        return {k: v.compute().cpu().item() for k, v in self.train_losses}

    @torch.no_grad
    def compute_val_metrics(self) -> dict[str, float]:
        return {k: v.compute().cpu().item() for k, v in self.val_losses}

    def _critic_step(
        self,
        model: GarmentConditionalGAN,
        data: TData,
        device: TDevice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = data
        signal = torch.randn(images.shape[0], self.signal_length, device=device)

        # Generate fake images.
        fake_images = model({"signal": signal, "class": labels}, network="generator")

        # Let the critic discriminates between the images.
        real_critic = model({"images": images, "class": labels}, network="critic")
        fake_critic = model({"images": fake_images, "class": labels}, network="critic")
        critic_loss = fake_critic - real_critic

        # Generate a mixture of fake and real images.
        eps = torch.rand(device=device)
        mixed_images = eps * images + (1. - eps) * fake_images

        # Calculate the gradient penalty term.

    def _generator_step(
        self,
        model: GarmentConditionalGAN,
        data: TData,
        device: TDevice,
    ) -> torch.Tensor:
        labels = data[1]
        batch_size = labels.shape[0]

        x: GeneratorInput = {
            "signal": torch.randn(batch_size, self.signal_length, device=device),
            "class": data[1],
        }
        fake_images = model(x, network="generator")

        x: DiscriminatorInput = {
            "image": fake_images,
            "class": data[1],
        }
        x = model(x, network="critic")

        return -torch.mean(x)
