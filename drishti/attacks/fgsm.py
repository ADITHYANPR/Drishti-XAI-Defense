import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon=0.03):
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    perturbation = epsilon * images.grad.sign()
    adversarial_images = images + perturbation

    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    return adversarial_images.detach()