# Synthetic Electronic Warfare Simulation Module
"""
Drishti-XAI-Defense
Synthetic Electronic Warfare Simulation Module

Simulates contested battlefield sensor degradation.
"""

import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.02):
    sp_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(amount * total_pixels / 2)
    num_pepper = int(amount * total_pixels / 2)

    # Salt
    coords = (
        np.random.randint(0, image.shape[0], num_salt),
        np.random.randint(0, image.shape[1], num_salt),
    )
    sp_image[coords] = 255

    # Pepper
    coords = (
        np.random.randint(0, image.shape[0], num_pepper),
        np.random.randint(0, image.shape[1], num_pepper),
    )
    sp_image[coords] = 0

    return sp_image


def add_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def add_fog(image, fog_intensity=0.4):
    fog = np.full_like(image, 255)
    foggy = cv2.addWeighted(image, 1 - fog_intensity, fog, fog_intensity, 0)
    return foggy


def apply_battlefield_degradation(
    image,
    gaussian=True,
    salt_pepper=True,
    motion=True,
    fog=True,
):
    degraded = image.copy()

    if gaussian:
        degraded = add_gaussian_noise(degraded)

    if salt_pepper:
        degraded = add_salt_pepper_noise(degraded)

    if motion:
        degraded = add_motion_blur(degraded)

    if fog:
        degraded = add_fog(degraded)

    return degraded
