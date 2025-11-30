import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data.data import (
    load_dataset,
    split_dataset,
    create_data_loaders,
)
from src.data.transforms import get_base_transforms, get_train_transforms
from src.data.dataset import TransformDataset
from src.model.architecture import create_model
from src.training.trainer import train_loop
from src.utils.device import get_device
from src.utils.seed import set_seed


def main(args):
    set_seed(42)
    device = get_device()

    # Load data
    dataset = load_dataset(args.data_dir)
    train_subset, test_subset, val_subset = split_dataset(
        dataset, args.test_split, args.val_split
    )

    train_dataset = TransformDataset(train_subset, transform=get_train_transforms())
    val_dataset = TransformDataset(val_subset, transform=get_base_transforms())
    test_dataset = TransformDataset(test_subset, transform=get_base_transforms())

    train_loader, test_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataset=dataset,
        balance=True,
        batch_size=args.batch_size,
    )

    # Model
    num_classes = len(dataset.classes)
    model = create_model(num_classes=num_classes, freeze_features=True).to(device)

    if args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")
        print(f"Loading model weights from {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print("✅ Model weights loaded successfully.")

    # Optimizer & Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Train
    train_loop(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        num_epochs=args.epochs,
        writer=writer,
    )

    writer.close()

    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument(
        "--data_dir", type=str, default="../data/raw/", help="Path to dataset"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument(
        "--balance", action="store_true", help="Use WeightedRandomSampler"
    )
    parser.add_argument("--log_dir", type=str, default="../experiments/run1")
    parser.add_argument("--save_path", type=str, default="../models/run1.pth")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a saved model (.pth) to resume training on",
    )

    args = parser.parse_args()
    main(args)
