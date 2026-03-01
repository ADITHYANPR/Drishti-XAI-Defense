import cv2
from drishti.ew_layer.battlefield_noise import apply_battlefield_degradation

# Correct path since image is inside assets folder
image = cv2.imread("assets/test_image.jpg")

if image is None:
    print("Image not found. Check file path.")
    exit()

image = cv2.resize(image, (256, 256))
degraded = apply_battlefield_degradation(image)

cv2.imwrite("original_output.jpg", image)
cv2.imwrite("degraded_output.jpg", degraded)

print("Saved original_output.jpg and degraded_output.jpg")