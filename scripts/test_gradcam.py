import torch
import cv2
import numpy as np
from torchvision import transforms

from drishti.models.resnet_backbone import DrishtiResNet
from drishti.xai.gradcam import GradCAM, overlay_heatmap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DrishtiResNet(num_classes=3).to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = True

# Use last convolutional layer
target_layer = model.model.layer4[-1]

gradcam = GradCAM(model, target_layer)

image = cv2.imread("assets/test_image.jpg")
image = cv2.resize(image, (224, 224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(image).unsqueeze(0).to(device)

cam = gradcam.generate(input_tensor)

heatmap_overlay = overlay_heatmap(image, cam)

cv2.imwrite("gradcam_output.jpg", heatmap_overlay)

print("Saved gradcam_output.jpg")