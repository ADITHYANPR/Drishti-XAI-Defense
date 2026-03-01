import torch
from drishti.models.resnet_backbone import DrishtiResNet
from drishti.utils.dataloader import get_dataloaders
from drishti.attacks.fgsm import fgsm_attack
from drishti.evaluation.metrics import calculate_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, classes = get_dataloaders()

model = DrishtiResNet(num_classes=len(classes)).to(device)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.eval()

clean_acc = calculate_accuracy(model, val_loader, device)

correct = 0
total = 0

for images, labels in val_loader:
    images = images.to(device)
    labels = labels.to(device)

    adv_images = fgsm_attack(model, images, labels, epsilon=0.03)

    outputs = model(adv_images)
    _, preds = torch.max(outputs, 1)

    correct += (preds == labels).sum().item()
    total += labels.size(0)

adv_acc = correct / total

print(f"Clean Accuracy: {clean_acc:.4f}")
print(f"Adversarial Accuracy: {adv_acc:.4f}")
print(f"Adversarial Robustness: {adv_acc / clean_acc if clean_acc > 0 else 0:.4f}")