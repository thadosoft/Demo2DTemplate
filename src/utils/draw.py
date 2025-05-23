import cv2
import numpy as np
def draw_rectangles(image, rectangles):
    """Draw rectangles on the image."""
    result = image.copy()
    if len(result.shape) == 2:  # Nếu ảnh gốc là grayscale
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    for i, rect in enumerate(rectangles):
        # Draw rectangle outline
        cv2.drawContours(result, [rect], 0, (0, 255, 255), 2)
        
        # Draw rectangle center
        M = cv2.moments(rect)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result, (cx, cy), 2, (0, 255, 255), -1)
            cv2.putText(result, f"R{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
    return result

def draw_circles(image, circles, pixel_per_cm = 100):
    """Draw circles on the image."""
    result = image.copy()
    for i, circle in enumerate(circles):
        center = circle['center']
        radius = circle['radius']
        # Draw circle outline
        cv2.circle(result, center, radius, (0, 0, 255), 2)
        # Draw center point
        cv2.circle(result, center, 1, (0, 0, 255), -1)
        cv2.putText(result, f"C{i+1} R={radius/pixel_per_cm:.3f}", (center[0]+int(radius/5), center[1]-int(radius/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

    return result

def find_closest_point_on_box(box, center, distance, inside):
    """
    Tìm điểm gần nhất trên box với tâm đường tròn.
    
    Args:
        box: Box của hình chữ nhật đã làm phẳng
        center: Tọa độ tâm đường tròn
        distance: Khoảng cách đã tính từ pointPolygonTest
        inside: True nếu tâm nằm trong hình chữ nhật, False nếu nằm ngoài
        
    Returns:
        closest_point: Tọa độ điểm gần nhất
    """
    # Chuyển đổi tọa độ tâm sang tuple int
    center_int = (int(center[0]), int(center[1]))
    
    # Tìm điểm gần nhất trên box
    min_dist = float('inf')
    closest_point = None
    
    # Kiểm tra các điểm trên box
    for point in box:
        point = tuple(point)
        dist = np.sqrt((point[0] - center_int[0])**2 + (point[1] - center_int[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_point = point
    
    # Kiểm tra các điểm trên các cạnh của box
    for k in range(len(box)):
        p1 = np.array(box[k], dtype=np.float32)
        p2 = np.array(box[(k+1) % len(box)], dtype=np.float32)
        center_arr = np.array(center_int, dtype=np.float32)
        
        edge = p2 - p1
        edge_length = np.linalg.norm(edge)
        if edge_length < 1e-6:
            continue
        
        t_vec = center_arr - p1
        t = np.dot(t_vec, edge) / (edge_length ** 2)
        t = max(0.0, min(1.0, t))
        
        closest_on_edge = p1 + t * edge
        curr_dist = np.linalg.norm(center_arr - closest_on_edge)
        
        if curr_dist < min_dist:
            min_dist = curr_dist
            closest_point = tuple(np.round(closest_on_edge).astype(int))
    
    # Nếu không tìm thấy điểm nào
    if closest_point is None:
        # Tính một điểm theo hướng bất kỳ từ tâm với khoảng cách đã biết
        angle = 0  # Chọn một góc, ví dụ 0 radian
        closest_point = (
            int(center_int[0] + distance * np.cos(angle)),
            int(center_int[1] + distance * np.sin(angle))
        )
    
    return closest_point

def draw_circle_center_to_rectangle_distances(image, circles, rectangles, distances, pixel_per_cm=100):
    """
    Draw the shortest distance from each circle center to the nearest rectangle edge on the image.
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    font_thickness = 0.01
    line_thickness = 1
    circle_center = 1
    for i, circle in enumerate(circles):
        if i == 0:
            font_scale = circle['radius'] * 0.025
    # print(font_scale)
        font_thickness = int(circle['radius'] * 0.1)
        line_thickness = int(circle['radius'] * 0.1)
        if font_thickness < 1:
            font_thickness = 1
        if line_thickness < 1:
            line_thickness = 1
        if font_scale == 0:
            font_scale = 0.01
        circle_center = circle['center']
        
        for j, contour in enumerate(rectangles):
            key = f"Circle {i} to Rectangle {j}"
            if key not in distances:
                continue
                
            distance_info = distances[key]
            distance = distance_info['distance']
            inside = distance_info['inside']
            box = distance_info['box']  # Lấy box đã tính toán trước
            
            # Tìm điểm gần nhất trên các cạnh của hình chữ nhật đã làm phẳng
            closest_point = None
            min_dist = float('inf')
            
            # Duyệt qua từng cạnh của hình chữ nhật
            for k in range(len(box)):
                # Lấy điểm đầu và cuối của cạnh
                p1 = np.array(box[k], dtype=np.float32)
                p2 = np.array(box[(k+1) % len(box)], dtype=np.float32)
                center = np.array(circle_center, dtype=np.float32)
                
                # Vector của cạnh
                edge = p2 - p1
                edge_length = np.linalg.norm(edge)
                if edge_length < 1e-6:  # Bỏ qua cạnh không hợp lệ
                    continue
                
                # Vector từ p1 đến tâm
                t_vec = center - p1
                
                # Tính hình chiếu của tâm lên cạnh
                t = np.dot(t_vec, edge) / (edge_length ** 2)
                t = max(0.0, min(1.0, t))  # Clamp về [0, 1]
                
                # Điểm gần nhất trên cạnh
                closest_on_edge = p1 + t * edge
                
                # Khoảng cách từ tâm đến điểm này
                current_dist = np.linalg.norm(center - closest_on_edge)
                
                # Cập nhật điểm gần nhất
                if current_dist < min_dist:
                    min_dist = current_dist
                    closest_point = closest_on_edge
            
            # Chuyển đổi tọa độ điểm về integer
            if closest_point is not None:
                closest_point = tuple(np.round(closest_point).astype(int))
                
                # Kiểm tra sai số khoảng cách
                if abs(min_dist - distance) > 2.0:
                    # Nếu có sai số lớn, tìm điểm gần nhất trên box
                    closest_point = find_closest_point_on_box(box, circle_center, distance, inside)
                
                # Màu sắc tùy thuộc vào vị trí
                color = (0, 0, 255) if inside else (0, 255, 255)
                
                # Vẽ đường thẳng
                cv2.line(result, circle_center, closest_point, color, line_thickness, cv2.LINE_AA)
                
                # Thêm mũi tên chỉ hướng
                dx = closest_point[0] - circle_center[0]
                dy = closest_point[1] - circle_center[1]
                angle = np.arctan2(dy, dx)
                arrow_length = 15
                
                # Tính toán các điểm mũi tên
                p1_arrow = (
                    int(closest_point[0] - arrow_length * np.cos(angle + np.pi/6)),
                    int(closest_point[1] - arrow_length * np.sin(angle + np.pi/6))
                )
                p2_arrow = (
                    int(closest_point[0] - arrow_length * np.cos(angle - np.pi/6)),
                    int(closest_point[1] - arrow_length * np.sin(angle - np.pi/6))
                )
                
                cv2.line(result, closest_point, p1_arrow, color, line_thickness)
                cv2.line(result, closest_point, p2_arrow, color, line_thickness)
                
                # Hiển thị giá trị khoảng cách
                mid_x = (circle_center[0] + closest_point[0]) // 2
                mid_y = (circle_center[1] + closest_point[1]) // 2
                text = f"{distance/pixel_per_cm:.1f}"
                
                # Vẽ nền cho chữ
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(result, 
                    (mid_x - tw//2 - 2, mid_y - th//2 - 2),
                    (mid_x + tw//2 + 2, mid_y + th//2 + 2),
                    (0, 0, 0), -1)
                
                # Vẽ chữ
                cv2.putText(result, text, (mid_x - tw//2, mid_y + th//2),
                          font, font_scale, (255, 255, 255), font_thickness)
    
    return result