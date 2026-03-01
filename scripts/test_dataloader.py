from drishti.utils.dataloader import get_dataloaders

train_loader, val_loader, classes = get_dataloaders()

print("Classes:", classes)
print("Number of training batches:", len(train_loader))