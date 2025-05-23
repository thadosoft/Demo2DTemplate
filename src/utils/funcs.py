import cv2
from datetime import datetime
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
import supervision as sv
import difflib

def maskImage(image, mask):
    # Chuyển ảnh mask thành ảnh xám
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Đảm bảo kiểu dữ liệu là CV_8U
    gray_mask = np.uint8(gray_mask)

    # Tạo mask nhị phân (đen trắng)
    ret, thresh = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Áp dụng mask lên ảnh gốc
    result = cv2.bitwise_and(image, image, mask=thresh)

    return result

def similarity_percentage(str1, str2):
    similarity_ratio = difflib.SequenceMatcher(None, str1, str2).ratio()
    return similarity_ratio * 100

def mask_image_by_box(img, a, b, c, d):
    white_img = np.ones_like(img) * 255

    mask = np.zeros_like(img)
    cv2.rectangle(mask, (a, b), (c, d), (255, 255, 255), thickness=cv2.FILLED)

    result = np.where(mask == np.array([255, 255, 255]), img, white_img)
    return result

def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    
    height, width = (img.shape[0], img.shape[1])
    result = []
    for seg in keypoints:
        keypoints_pixels = []
        for i in range(0, len(seg), 2):
            x = seg[i] * width
            y = seg[i + 1] * height
            keypoints_pixels.append((x, y))
        result.append(keypoints_pixels)
    
    for seg in result:
        pts = np.array([seg], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return img

def display_fps(img, start_time):
    end_time = datetime.now()
    fps = 1 / np.round((end_time - start_time).total_seconds(), 2)
    text = f"FPS: {int(fps)}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    gap = 10
    cv2.rectangle(
        img,
        (20 - gap, 170 - text_size[1] - gap),
        (20 + text_size[0] + gap, 170 + gap),
        (255, 255, 255),
        -1,
    )
    cv2.putText(img, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return fps

def crop_image(img, x, y, width, height):
    cropped_image = np.copy(img[y:y+height, x:x+width])
    #cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    return cropped_image

def crop_image_by_box(img, box, width=None, height=None):
    if not width and not height:
        xb1, yb1, xb2, yb2 = box[0], box[1], box[2], box[3]
    else:
        xb1, yb1, xb2, yb2 = box[0], box[1], box[2], box[3]
        xb1 = int(xb1 * width)
        xb2 = int(xb2 * width)
        yb1 = int(yb1 * height)
        yb2 = int(yb2 * height)
    return img[yb1:yb2, xb1:xb2]

def read_coco_segmentation(file_path):
    with open(file_path, 'r') as f:
        s = f.read()
        l = [line.split(' ') for line in s.split('\n')]
        l = [[float(element) for element in sublist if element] for sublist in l]
        l = [sublist[1:] for sublist in l]
        print(l)
    return l

def keypoints_to_pixels(keypoints, image_shape):
    height, width = image_shape[:2]
    result = []
    for seg in keypoints:
        keypoints_pixels = []
        for i in range(0, len(seg), 2):
            x = seg[i] * width
            y = seg[i + 1] * height
            keypoints_pixels.append((x, y))
        result.append(keypoints_pixels)
    return result

def keypoints_to_mask(image_shape, keypoints_pixels):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for seg in keypoints_pixels:
        pts = np.array(seg, np.int32)
        cv2.fillPoly(mask, [pts], (255))
    return mask

def draw_rect(img, x1, y1, x2, y2, text):
    color = (255, 0, 0)
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img

def save_image(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)

def convert_resolution(box, from_reso, to_reso):
    xb1, yb1, xb2, yb2 = box[0], box[1], box[2], box[3]
    xb1 = int(xb1 * to_reso[0] / from_reso[0])
    xb2 = int(xb2 * to_reso[0] / from_reso[0])
    yb1 = int(yb1 * to_reso[1] / from_reso[1])
    yb2 = int(yb2 * to_reso[1] / from_reso[1])
    return (xb1, yb1, xb2, yb2)

def matchTemplate(img, template,threshold=0.80):
    (tH, tW) = template.shape[:2]
    rects = []
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    (yCoords, xCoords) = np.where(res >= threshold)
    for (x, y) in zip(xCoords, yCoords):
        # update our list of rectangles
        rects.append((x, y, x + tW, y + tH))
    pick = non_max_suppression_fast(np.array(rects), 0.1)
    return pick

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
    
