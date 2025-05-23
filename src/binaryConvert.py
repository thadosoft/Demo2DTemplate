import cv2


def apply_canny(img):
    # img = cv2.resize(img, (1290, 960))
    # img = cv2.GaussianBlur(img, (9, 9), 0)

    can = cv2.Canny(img, 255, 255)

    countour,_ = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, countour, -1, (255, 0, 0), 2)
    return can

