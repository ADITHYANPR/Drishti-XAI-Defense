# Training Entry Script
import torch
import torch.nn as nn
import torch.optim as optim

from drishti.models.resnet_backbone import DrishtiResNet
from drishti.utils.dataloader import get_dataloaders
from drishti.evaluation.metrics import calculate_accuracy
from drishti.ew_layer.battlefield_noise import apply_battlefield_degradation


from drishti.attacks.fgsm import fgsm_attack


def train_one_epoch(model, loader, criterion, optimizer, device, epsilon=0.03):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # --- Clean Forward ---
        optimizer.zero_grad()
        outputs = model(images)
        loss_clean = criterion(outputs, labels)

        # --- Adversarial Forward ---
        adv_images = fgsm_attack(model, images.clone(), labels, epsilon=epsilon)
        outputs_adv = model(adv_images)
        loss_adv = criterion(outputs_adv, labels)

        # --- Combined Loss ---
        loss = (loss_clean + loss_adv) / 2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate_with_noise(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images_np = images.permute(0, 2, 3, 1).cpu().numpy()

            degraded_images = []
            for img in images_np:
                img_uint8 = (img * 255).astype("uint8")
                degraded = apply_battlefield_degradation(img_uint8)
                degraded = torch.tensor(degraded / 255.0, dtype=torch.float32).permute(2, 0, 1)
                degraded_images.append(degraded)

            degraded_images = torch.stack(degraded_images).to(device)
            labels = labels.to(device)

            outputs = model(degraded_images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_dataloaders()

    model = DrishtiResNet(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_acc = 0.0

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        clean_acc = calculate_accuracy(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Loss: {loss:.4f}")
        print(f"Validation Accuracy: {clean_acc:.4f}")

        if clean_acc > best_acc:
           best_acc = clean_acc
           torch.save(model.state_dict(), "best_model.pth")
           print("Best model saved.")

    print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()