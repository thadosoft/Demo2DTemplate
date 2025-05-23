import cv2
import numpy as np

def apply_canny(img):
    # img = cv2.resize(img, (1290, 960))
    # img = cv2.GaussianBlur(img, (9, 9), 0)

    can = cv2.Canny(img, 255, 255)

    countour,_ = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, countour, -1, (255, 0, 0), 2)
    return can

def preprocessed_image(image):
    # Image is already grayscale, no need to convert
    blurred = cv2.blur(image, (5,5))
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + 2)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl1 = clahe.apply(blurred)
    _, binary_2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(binary_2, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
