import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import sys


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=False,
        ncols=100,
        file=sys.stdout,
    )

    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


def validate(model, dataloader, loss_fn, device, epoch, num_epochs):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs} [Val]",
        leave=False,
        ncols=100,
        file=sys.stdout,
    )

    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)

            total_loss += loss.item()
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", ncols=120)

    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, num_epochs
        )

        val_metrics = validate(model, val_loader, loss_fn, device, epoch, num_epochs)

        epoch_bar.set_postfix(
            {
                "train_loss": f"{train_metrics[0]:.4f}",
                "val_loss": f"{val_metrics[0]:.4f}",
                "val_acc": f"{val_metrics[1]:.4f}",
                "val_f1": f"{val_metrics[2]:.4f}",
            }
        )

        tqdm.write(
            f"Epoch {epoch:3d}/{num_epochs} │ "
            f"Train: Loss={train_metrics[0]:.4f} Acc={train_metrics[1]:.4f} F1={train_metrics[2]:.4f} │ "
            f"Val: Loss={val_metrics[0]:.4f} Acc={val_metrics[1]:.4f} F1={val_metrics[2]:.4f}"
        )

        for key, val in zip(["train_loss", "train_acc", "train_f1"], train_metrics):
            history[key].append(val)
        for key, val in zip(["val_loss", "val_acc", "val_f1"], val_metrics):
            history[key].append(val)

        if writer:
            writer.add_scalar("Loss/Train", train_metrics[0], epoch)
            writer.add_scalar("Loss/Val", val_metrics[0], epoch)
            writer.add_scalar("Accuracy/Train", train_metrics[1], epoch)
            writer.add_scalar("Accuracy/Val", val_metrics[1], epoch)
            writer.add_scalar("F1/Train", train_metrics[2], epoch)
            writer.add_scalar("F1/Val", val_metrics[2], epoch)

    epoch_bar.close()
    return history
