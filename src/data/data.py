import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler


def get_sample_weights(dataset, train_dataset):
    # Code taken from:
    #     https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
    y_train_indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in y_train_indices]

    class_sample_counts = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    )

    weights = 1.0 / class_sample_counts
    sample_weights = np.array([weights[t] for t in y_train])
    sample_weights = torch.from_numpy(sample_weights)

    return sample_weights


def load_dataset(data_dir, transform=None):
    return datasets.ImageFolder(root=data_dir, transform=transform)


def split_dataset(dataset, test_split=0.15, val_split=0.1):
    total = len(dataset)
    test_len = int(test_split * total)
    val_len = int(val_split * total)
    train_len = total - test_len - val_len
    return random_split(dataset, [train_len, test_len, val_len])


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    dataset,
    batch_size=128,
    balance=True,
    sampling_strategy=5000,
):
    if balance:
        sample_weights = get_sample_weights(dataset, train_dataset)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True if train_sampler else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
