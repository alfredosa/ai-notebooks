#!/usr/bin/env python3

import torch
from torchvision import transforms, datasets

print("Cuda available: ", torch.cuda.is_available())

transform = transforms.Compose([
    # Because the MNIST Dataset is basically a PIL Image
    transforms.ToTensor(),


    # Formula here: output[channel] = (input[channel] - mean[channel]) / std[channel]
    # Idea is to basically optimize the data, big numbers cause the training to
    # basically get
    # - Slower Convergence
    # - Getting stuck in local minima
    # - sensitivity of learning rate
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

datasets.MNIST(root="./data", train=True, download=True, transform=)
