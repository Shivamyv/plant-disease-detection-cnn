import argparse
from pathlib import Path
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/plantvillage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    root = Path(args.data)
    if not (root / "train").exists() or not (root / "val").exists():
        raise FileNotFoundError("Expected train/ and val/ folders under --data.")
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train and val class folders must match exactly.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc, best_state = 0.0, copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_accuracy": []}
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        history["train_loss"].append(total_loss / max(len(train_ds), 1))
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={history['train_loss'][-1]:.4f}, val_accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    Path("models").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, "models/best_model.pt")
    print(f"Saved model. Best validation accuracy: {best_acc:.4f}")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_dl:
            preds = model(images.to(device)).argmax(1).cpu()
            y_true.extend(labels.tolist()); y_pred.extend(preds.tolist())
    print(classification_report(y_true, y_pred, target_names=train_ds.classes, zero_division=0))
    print("Note: these are validation results; evaluate once on a separate test set for final reporting.")
    Path("reports").mkdir(exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(history["train_loss"])+1), history["train_loss"], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Training loss")
    plt.tight_layout(); plt.savefig("reports/training_loss.png", dpi=160)

if __name__ == "__main__":
    main()
