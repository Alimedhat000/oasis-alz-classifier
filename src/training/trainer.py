# src/training/trainer.py
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)

            total_loss += loss.item()
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


def train_loop(
    model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, writer=None
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate(model, val_loader, loss_fn, device)

        # Log
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.4f}, F1: {train_metrics[2]:.4f} | "
            f"Val Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.4f}, F1: {val_metrics[2]:.4f}"
        )

        # Save
        for key, val in zip(["train_loss", "train_acc", "train_f1"], train_metrics):
            history[key].append(val)
        for key, val in zip(["val_loss", "val_acc", "val_f1"], val_metrics):
            history[key].append(val)

        # TensorBoard (optional)
        if writer:
            writer.add_scalar("Loss/Train", train_metrics[0], epoch)
            writer.add_scalar("Loss/Val", val_metrics[0], epoch)
            writer.add_scalar("Accuracy/Train", train_metrics[1], epoch)
            writer.add_scalar("Accuracy/Val", val_metrics[1], epoch)
            writer.add_scalar("F1/Train", train_metrics[2], epoch)
            writer.add_scalar("F1/Val", val_metrics[2], epoch)

    return history
