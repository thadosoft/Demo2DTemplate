import cv2
import numpy as np
def check_circle_in_rectangle(result_image, circles, rectangles):
    is_ok = True
    
    for contour in rectangles:
        rect = cv2.minAreaRect(contour)
        
        (center_x, center_y), (w, h), angle = rect
        # print(w, h, angle)
        if w < h:
            print(w, h, angle)
            if angle > 80 and angle < 90:
                angle -= 90
            else:
                angle += 90
            temp = w
            w = h
            h = temp
        
        # Vẽ các đường tham chiếu (cách mép 0.2 chiều rộng/chiều cao)
        offset_x1 = int(0.125 * w)
        offset_x2 = int(0.122 * w)
        offset_y1 = int(0.25 * h)
        offset_y2 = int(0.225 * h)

        # Ma trận xoay ngược góc để đưa offset về tọa độ ảnh gốc
        rotation_matrix = cv2.getRotationMatrix2D((0, 0), -angle, 1)

        # Danh sách các điểm cần tính (tọa độ tương đối trong hình chữ nhật)
        relative_points = [
            (-w/2 + offset_x1, -h/2 + offset_y1),           # Góc trên trái
            (w/2 - offset_x1, -h/2 + offset_y1),            # Góc trên phải
            (-w/2 + offset_x2, h/2 - offset_y2),            # Góc dưới trái
            (w/2 - offset_x2, h/2 - offset_y2),             # Góc dưới phải
            (0, h/2 - offset_y2)                           # Trung tâm
        ]

        # Chuyển điểm tương đối thành tọa độ ảnh
        expected_positions = []
        for dx, dy in relative_points:
            rotated = rotation_matrix @ np.array([[dx], [dy], [1]])
            ex, ey = rotated[0][0] + center_x, rotated[1][0] + center_y
            expected_positions.append((ex, ey))

        
        # Kiểm tra mỗi hình tròn có nằm đúng vị trí không
        correct_positions = [False] * len(expected_positions)

        for circle in circles:
            border_radius = int(circle['radius'] + w * 0.02)
            cx, cy = circle['center']
            tolerance = abs(circle['radius'] - border_radius)
            # print(tolerance)
            # Kiểm tra với mỗi vị trí mong đợi
            for i, (ex, ey) in enumerate(expected_positions):
                distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                # print(distance)
                if distance < tolerance:
                    correct_positions[i] = True
        
        # Hiển thị kết quả kiểm tra
        for i, correct in enumerate(correct_positions):
            position_name = ["Góc trên trái", "Góc trên phải", "Góc dưới trái", "Góc dưới phải", "Trung tâm"][i]
            status = "Đúng" if correct else "Thiếu"
            # print(f"{position_name}: {status}")
            
            # Nếu vị trí thiếu, đổi hình tròn thành màu đỏ
            ex, ey = expected_positions[i]
            if not correct:
                is_ok = False
                cv2.circle(result_image, (int(ex), int(ey)), border_radius, (255, 0, 0), 3)
            else:
                cv2.circle(result_image, (int(ex), int(ey)), border_radius, (0, 255, 0), 3)
    return result_image, is_ok


# Main function to process the image

def calculate_circle_center_to_rectangle_distances(circles, rectangles):
    """
    Calculate the distance from each circle center to the nearest edge of each rectangle
    using cv2.pointPolygonTest which gives the shortest distance.
    
    Args:
        circles: List of detected circle information
        rectangles: List of detected rectangle contours
        
    Returns:
        distances: Dictionary with circle and rectangle indices as keys and distances as values
    """
    distances = {}
    
    for i, circle in enumerate(circles):
        circle_center = circle['center']
        
        for j, contour in enumerate(rectangles):
            # Calculate distance from circle center to rectangle using pointPolygonTest
            # This returns negative distance if outside, positive if inside, and 0 if on the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Sửa từ np.intp sang np.intp để đảm bảo tương thích
            
            # Quan trọng: Chuyển box thành contour để tính khoảng cách
            box_contour = box.reshape(-1, 1, 2)
            dist = cv2.pointPolygonTest(box_contour, circle_center, True)  # Tính trên box đã làm phẳng
            
            # Always use absolute value to get the actual distance
            distance = abs(dist)
            
            # Store the distance and whether the point is inside or outside
            key = f"Circle {i} to Rectangle {j}"
            distances[key] = {
                'distance': distance,
                'inside': dist > 0,
                'circle_center': circle_center,
                'rect': rect,
                'box': box
            }
    
    return distances



