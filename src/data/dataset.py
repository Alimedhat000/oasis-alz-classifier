from torch.utils.data import Dataset


class TransformDataset(Dataset):
    """
    Wrapper class to assign a transform to a subset.
    Forwards .indices if the underlying subset has it (e.g., from random_split).
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def indices(self):
        # Forward .indices if available (e.g., when subset is a torch.utils.data.Subset)
        if hasattr(self.subset, "indices"):
            return self.subset.indices
        else:
            # Optional: raise or return None if not applicable
            raise AttributeError("'subset' has no attribute 'indices'")
