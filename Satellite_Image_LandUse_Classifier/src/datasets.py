from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    return train_transform, val_transform


def get_dataloaders(
    data_dir,
    batch_size=32,
    train_split=0.8,
    seed=42
):

    train_transform, val_transform = get_transforms()

    full_dataset = datasets.ImageFolder(data_dir)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=train_transform),
        train_subset.indices
    )

    val_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=val_transform),
        val_subset.indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        full_dataset.classes,
        full_dataset.class_to_idx
    )