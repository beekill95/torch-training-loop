# %%
# %cd ..

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torcheval.metrics import BinaryAccuracy
import torchvision
import torchvision.transforms as transforms
from training_loop import TrainingLoop
from typing import Literal, Tuple, TypedDict

# %% [markdown]
# # GAN on Fashion MNIST with Custom Training Loop
# ## Preparing the Data

# %%
# Transform the images to range (-1, 1).
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

# %% [markdown]
# ## GAN Models: Generator and Discriminator
#
# Similar to [this example.](https://keras.io/examples/generative/conditional_gan/#creating-the-discriminator-and-generator)

# %%
GeneratorInput = TypedDict('GeneratorInput', {
    'signal': torch.Tensor,
    'class': torch.Tensor,
})
DiscriminatorInput = TypedDict('DiscriminatorInput', {
    'image': torch.Tensor,
    'class': torch.Tensor,
})


class GarmentGenerator(nn.Module):

    def __init__(self, input_size: int, n_classes: int) -> None:
        super().__init__()

        n_channels = 7
        self.linear = nn.Sequential(
            nn.Linear(input_size + n_classes, 7 * 7 * n_channels),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_channels,
                               128,
                               kernel_size=(4, 4),
                               stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=(7, 7)),
            nn.Tanh(),
        )

    def forward(self, input: GeneratorInput):
        signal, clazz = input['signal'], input['class']
        x = torch.cat([signal, clazz], dim=-1)
        batch_size = x.shape[0]

        x = self.linear(x)
        x = x.view((batch_size, -1, 7, 7))
        x = self.conv(x)
        return x


class GarmentDiscriminator(nn.Module):

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.output = nn.Linear(128 + n_classes, 1)

    def forward(self, input: DiscriminatorInput):
        image, clazz = input['image'], input['class']
        batch_size = image.shape[0]

        x = self.conv(image)
        x = x.view((batch_size, -1))

        x = torch.cat([x, clazz], dim=-1)
        x = self.output(x)

        return x


class GarmentConditionalGAN(nn.Module):

    def __init__(self, signal_length: int, n_classes: int) -> None:
        super().__init__()

        self.generator = GarmentGenerator(signal_length, n_classes)
        self.discriminator = GarmentDiscriminator(n_classes)

    def forward(
        self,
        input: GeneratorInput | DiscriminatorInput,
        network: Literal['generator', 'discriminator'] = 'generator',
    ):
        if network == 'generator':
            return self.generator(input)
        elif network == 'discriminator':
            return self.discriminator(input)

        raise ValueError(f'Unknown network={network}!')


# %% [markdown]
# ## Training
# ### Custom Training Loop


# %%
class GarmentConditionalGANTrainingLoop(TrainingLoop[GarmentConditionalGAN,
                                                     Tuple[torch.Tensor,
                                                           torch.Tensor]]):

    def __init__(self,
                 model: GarmentConditionalGAN,
                 *,
                 signal_length: int,
                 n_classes: int,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__(model, device=device)

        self.generator_optim = Adam(model.generator.parameters(), lr=0.0001)
        self.discriminator_optim = Adam(model.discriminator.parameters(),
                                        lr=0.0001)

        self.signal_length = signal_length
        self.n_classes = n_classes

        # Train metrics and losses.
        self.generator_total_train_loss = 0.
        self.generator_nb_train_batches = 0
        self.discriminator_total_train_loss = 0.
        self.discriminator_nb_train_batches = 0
        self.generator_train_accuracy = BinaryAccuracy(threshold=0.,
                                                       device=device)
        self.discriminator_train_accuracy = BinaryAccuracy(threshold=0.,
                                                           device=device)

        # Validation metrics and losses.
        self.generator_total_val_loss = 0.
        self.generator_nb_val_batches = 0
        self.discriminator_total_val_loss = 0.
        self.discriminator_nb_val_batches = 0
        self.generator_val_accuracy = BinaryAccuracy(threshold=0.,
                                                     device=device)
        self.discriminator_val_accuracy = BinaryAccuracy(threshold=0.,
                                                         device=device)

    def train_step(
            self, data: Tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        images, labels = data
        images = images.to(self.device)
        labels = F.one_hot(labels, self.n_classes).to(self.device)

        # First, perform the forward pass on the generator to obtain
        # fake images and generator_loss.
        (
            generator_loss,
            fake_images,
            discriminator_output,
        ) = self._generator_forward(images, labels)

        # Train the generator.
        self.generator_optim.zero_grad()
        generator_loss.backward()
        self.generator_optim.step()

        # Update the generator accuracy.
        with torch.no_grad():
            self.generator_train_accuracy.update(
                discriminator_output.squeeze(),
                torch.tensor([0.] * discriminator_output.shape[0],
                             device=self.device))

        # Next, perform the forward pass on the discriminator to
        # the discriminator's loss.
        (
            discriminator_loss,
            discriminator_output,
            discriminator_target,
        ) = self._discriminator_forward(images=images,
                                        labels=labels,
                                        fake_images=fake_images.detach(),
                                        fake_labels=labels)

        # Train the discriminator.
        self.discriminator_optim.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optim.step()

        # Update the losses.
        self.generator_total_train_loss += generator_loss.detach().cpu().item()
        self.generator_nb_train_batches += 1
        self.discriminator_total_train_loss += discriminator_loss.detach().cpu(
        ).item()
        self.discriminator_nb_train_batches += 1

        with torch.no_grad():
            self.discriminator_train_accuracy.update(discriminator_output.squeeze(),
                                                     discriminator_target.squeeze())

        return self.compute_train_metrics()

    @torch.no_grad
    def val_step(self, data: Tuple[torch.Tensor,
                                   torch.Tensor]) -> dict[str, float]:
        self.model.eval()
        images, labels = data
        images = images.to(self.device)
        labels = F.one_hot(labels, self.n_classes).to(self.device)

        # First, perform the forward pass on the generator to obtain
        # fake images and generator_loss.
        (
            generator_loss,
            fake_images,
            discriminator_output,
        ) = self._generator_forward(images, labels)

        # Update the generator accuracy.
        self.generator_train_accuracy.update(
            discriminator_output.squeeze(),
            torch.tensor([0.] * discriminator_output.shape[0],
                         device=self.device))

        # Next, perform the forward pass on the discriminator to
        # the discriminator's loss.
        (
            discriminator_loss,
            discriminator_output,
            discriminator_target,
        ) = self._discriminator_forward(images=images,
                                        labels=labels,
                                        fake_images=fake_images.detach(),
                                        fake_labels=labels)

        # Update the losses.
        self.generator_total_val_loss += generator_loss.detach().cpu().item()
        self.generator_nb_val_batches += 1
        self.discriminator_total_val_loss += discriminator_loss.detach().cpu(
        ).item()
        self.discriminator_nb_val_batches += 1

        self.discriminator_val_accuracy.update(discriminator_output.squeeze(),
                                                discriminator_target.squeeze())

        return self.compute_val_metrics()

    def reset_train_metrics(self):
        self.generator_total_train_loss = 0.
        self.generator_nb_train_batches = 0
        self.discriminator_total_train_loss = 0.
        self.discriminator_nb_train_batches = 0

        self.generator_train_accuracy.reset()
        self.discriminator_train_accuracy.reset()

    def reset_val_metrics(self):
        self.generator_total_val_loss = 0.
        self.generator_nb_val_batches = 0
        self.discriminator_total_val_loss = 0.
        self.discriminator_nb_val_batches = 0

        self.generator_val_accuracy.reset()
        self.discriminator_val_accuracy.reset()

    @torch.no_grad()
    def compute_train_metrics(self) -> dict[str, float]:
        return {
            'generator_loss':
            self.generator_total_train_loss / self.generator_nb_train_batches,
            'discriminator_loss': self.discriminator_total_train_loss /
            self.discriminator_nb_train_batches,
            'generator_acc': self.generator_train_accuracy.compute().cpu().item(),
            'discriminator_acc': self.discriminator_train_accuracy.compute().cpu().item(),
        }

    @torch.no_grad()
    def compute_val_metrics(self) -> dict[str, float]:
        return {
            'generator_loss':
            self.generator_total_val_loss / self.generator_nb_val_batches,
            'discriminator_loss': self.discriminator_total_val_loss /
            self.discriminator_nb_val_batches,
            'generator_acc': self.generator_val_accuracy.compute().cpu().item(),
            'discriminator_acc': self.discriminator_val_accuracy.compute().cpu().item(),
        }

    def _generator_forward(self, images: torch.Tensor, labels: torch.Tensor):
        batch_size = images.shape[0]

        # First, generate random signal to feed into the generator.
        signal = torch.randn((batch_size, self.signal_length),
                             dtype=images.dtype,
                             device=self.device)
        generator_input: GeneratorInput = {'signal': signal, 'class': labels}

        # Generate fake images from the generator.
        fake_images = self.model(generator_input, network='generator')

        # Feed these fake images into the discriminator.
        discriminator_input: DiscriminatorInput = {
            'image': fake_images,
            'class': labels,
        }
        discriminator_output = self.model(discriminator_input,
                                          network='discriminator')

        # Calculate the generator loss.
        generator_loss = -discriminator_output.mean()

        return generator_loss, fake_images, discriminator_output

    def _discriminator_forward(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        fake_images: torch.Tensor,
        fake_labels: torch.Tensor,
    ):
        nb_images = images.shape[0]
        nb_fake_images = fake_images.shape[0]

        discriminator_input: DiscriminatorInput = {
            'image': torch.cat([images, fake_images], dim=0),
            'class': torch.cat([labels, fake_labels], dim=0),
        }
        discriminator_target = torch.tensor([[1.]] * nb_images +
                                            [[0.]] * nb_fake_images,
                                            device=self.device)

        discriminator_output = self.model(discriminator_input,
                                          network='discriminator')

        # Calculate discriminator loss.
        loss = F.binary_cross_entropy_with_logits(discriminator_output,
                                                  discriminator_target)

        return loss, discriminator_output, discriminator_target


model = GarmentConditionalGAN(128, len(classes))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loop = GarmentConditionalGANTrainingLoop(
    model,
    signal_length=128,
    n_classes=len(classes),
    device=device,
)
loop.fit(
    training_loader,
    validation_loader,
    epochs=1,
)

# %% [markdown]
# ## Samples from the Generator

# %%
model.eval()
n_images_per_class = 5

fig, axes = plt.subplots(
    nrows=len(classes),
    ncols=n_images_per_class,
    figsize=(4 * n_images_per_class, 4 * len(classes)),
)
for i in range(len(classes)):
    with torch.no_grad():
        clazz = F.one_hot(torch.tensor([i] * n_images_per_class),
                          num_classes=len(classes)).to(device)
        signal = torch.randn((n_images_per_class, 128), device=device)

        # Feed into the generator.
        input: GeneratorInput = {'class': clazz, 'signal': signal}
        fake_images = model(input, network='generator')
        fake_images = fake_images.cpu()

    # Show these fake images.
    for ax, img in zip(axes[i], fake_images):
        ax.imshow(img[0], cmap='gray')

        # Don't show the ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the class name.
    axes[i][0].set_ylabel(classes[i])

fig.tight_layout()
