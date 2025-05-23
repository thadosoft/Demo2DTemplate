import cv2
import numpy as np
from copy import deepcopy
import supervision as sv

from src.binaryConvert import preprocessed_image
from src.utils.draw import *
# from src.UL.ULObjectDetection import ULObjectDetection
from src.utils.funcs import *

class Monitor:
    def __init__(self) -> None:
        # Anomaly
        self.model = None
        self.pixel_per_cm = 0
        self.real_height_cm = 20
        self.real_width_cm = 30

    def calc_cm_in_image(self, real_height_cm, real_width_cm, rectangles):
        pixel_to_cm_ratio = 10
        for rectangle_contour in rectangles:
            # Tính bounding rectangle từ contour
            rect = cv2.minAreaRect(rectangle_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Lấy chiều dài và chiều rộng của hình chữ nhật trong ảnh (đơn vị pixel)
            # rect trả về ((cx, cy), (w, h), angle)
            width_pixel = rect[1][0]
            height_pixel = rect[1][1]
            
            # Đảm bảo chiều dài lớn hơn chiều rộng
            if width_pixel < height_pixel:
                width_pixel, height_pixel = height_pixel, width_pixel
            
            # Tính tỷ lệ pixel/cm theo cả chiều dài và chiều rộng
            width_ratio = width_pixel / real_width_cm
            height_ratio = height_pixel / real_height_cm
            
            # Lấy giá trị trung bình để có tỷ lệ chính xác hơn
            pixel_to_cm_ratio = (width_ratio + height_ratio) / 2
            break
        # print(pixel_to_cm_ratio)
        return pixel_to_cm_ratio
    def process(self, img):
        try:
            is_ok = True
            result = img.copy()
            # Create a grayscale copy for contour detection
            gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
            contours = preprocessed_image(gray_img)
            circles, rectangles = self.Detect(contours)
            if len(rectangles) != 1:
                is_ok = False
            else:
                self.pixel_per_cm = self.calc_cm_in_image(self.real_height_cm, self.real_width_cm, rectangles)
                result = draw_rectangles(result, rectangles)
                result = draw_circles(result, circles, self.pixel_per_cm)

                # Calc and draw distance between circles and rectangles
                # result, is_ok = check_circle_in_rectangle(result, circles, rectangles)
                # distances = calculate_circle_center_to_rectangle_distances(circles, rectangles)
                # result = draw_circle_center_to_rectangle_distances(result, circles, rectangles, distances, self.pixel_per_cm)
                
            if is_ok:
                cv2.putText(result, "OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)
            else:
                cv2.putText(result, "NG", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3, cv2.LINE_AA)
            
            return result
        except Exception as e:
            print(e)
            cv2.putText(img, "NG", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3, cv2.LINE_AA)
            return img
    
    
    def Detect(self, contours, min_area=100, duplicate_distance_threshold=10):
        # Write your code here
        circles = []
        rectangles = []
        max_area = 0
        for contour in contours:
            if cv2.contourArea(contour) > max_area:
                max_area = cv2.contourArea(contour) 
        for contour in contours:
            
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
                
            # Check for circle using circularity metric
            circularity = (4 * np.pi * area) / (perimeter**2)
            # print(circularity)
            if circularity > 0.6:
                # Fit a circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                # radius = int(radius)
                
                # print(2*area/perimeter)
                # Check if area ratio indicates a good circle fit
                fitted_circle_area = np.pi * radius * radius
                area_ratio = area / fitted_circle_area if fitted_circle_area > 0 else 0
                radius_1 = int(2*area/perimeter)
                radius_2 = int(perimeter/2/np.pi)
                radius_3 = int(np.sqrt(area/np.pi))
                radius = int(np.mean([radius_1, radius_2, radius_3]))
                # print(radius)
                # print(area_ratio)
                if 0.89 < area_ratio < 1.2:
                    # Check for duplicate circles
                    is_duplicate = False
                    for existing_circle in circles:
                        ex_center = existing_circle['center']
                        distance = np.sqrt((ex_center[0] - center[0])**2 + (ex_center[1] - center[1])**2)
                        radius_diff = abs(existing_circle['radius'] - radius)
                        
                        if distance < duplicate_distance_threshold and radius_diff < 5:
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        circle_info = {
                            'center': center,
                            'radius': radius,
                            'contour': contour
                        }
                        circles.append(circle_info)
                    continue

        #     if cv2.contourArea(contour) == max_area:
        #         epsilon = 0.0001 * cv2.arcLength(contour, True)
        #         smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
        #         rectangles.append(smooth_contour)
        # return circles, rectangles
            
            if area < max_area - 1000:
                continue
            # If not a circle, check for rectangle
            # Calculate distances from centroid to contour points
            distances = []
            for point in contour:
                x, y = point[0]
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                distances.append((distance, (x, y)))
            
            # Find min and max distances
            distances.sort()
            min_distance, min_point = distances[0]
            max_distance, max_point = distances[-1]
            
            # Find the point with minimum slope product (perpendicular to min_point)
            min_slope_value = float('inf')
            min_slope_point = None
            min_slope_distance = 0
            
            for point in contour:
                x, y = point[0]
                
                # Skip cases where we'd divide by zero
                if x == cx or min_point[0] == cx:
                    continue
                    
                slope = (y - cy) / (x - cx)
                min_slope = (min_point[1] - cy) / (min_point[0] - cx)
                
                # Calculate value to find most perpendicular point
                value = abs(slope * min_slope + 1)
                if value < min_slope_value:
                    min_slope_value = value
                    min_slope_point = (x, y)
                    min_slope_distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Calculate rectangle metrics
            if min_distance > 0 and area > 0:
                # Area ratio
                rs = abs((area - min_distance * min_slope_distance) / area)
            else:
                rs = float('inf')
                
            if max_distance > 0:
                # Diagonal length ratio
                diagonal = np.sqrt(min_distance**2 + min_slope_distance**2)
                rl = abs((max_distance - diagonal) / max_distance)
            else:
                rl = float('inf')
            
            # Check if contour is likely a rectangle
            if rs <= 0.8 and rl <= 0.1:
                # Check for duplicate rectangles
                is_duplicate = False
                for existing_rect in rectangles:
                    existing_M = cv2.moments(existing_rect)
                    if existing_M['m00'] == 0:
                        continue
                        
                    ex_cx = int(existing_M['m10'] / existing_M['m00'])
                    ex_cy = int(existing_M['m01'] / existing_M['m00'])
                    
                    # Check if centroids are close
                    distance = np.sqrt((ex_cx - cx)**2 + (ex_cy - cy)**2)
                    if distance < duplicate_distance_threshold:
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
                    rectangles.append(smooth_contour)
                    # rectangles.append(contour)
        
        return circles, rectangles


    def Calc_distance(self, img):
        # Write your code here
        return img
    
    def preprocess(self, img):
        pass