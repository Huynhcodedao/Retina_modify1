import torch
import numpy as np
import matplotlib.pyplot as plt
from model.anchor import Anchors

def test_larger_anchors():
    """
    Test the new anchor generation with larger anchors
    """
    print("Tạo anchor với tham số mới...")
    
    # Tạo anchors với tham số mới
    anchors = Anchors()
    anchor_boxes = anchors.forward()
    
    # Chuyển sang numpy để dễ xử lý
    anchor_boxes_np = anchor_boxes.numpy()
    
    # Tính toán chiều rộng, chiều cao và diện tích
    widths = anchor_boxes_np[:, 2] - anchor_boxes_np[:, 0]
    heights = anchor_boxes_np[:, 3] - anchor_boxes_np[:, 1]
    areas = widths * heights
    
    # In ra thống kê
    print(f"Tổng số anchor: {len(anchor_boxes_np)}")
    print(f"Phạm vi chiều rộng: {widths.min():.6f} đến {widths.max():.6f}")
    print(f"Phạm vi chiều cao: {heights.min():.6f} đến {heights.max():.6f}")
    print(f"Phạm vi diện tích: {areas.min():.6f} đến {areas.max():.6f}")
    
    # Tính phân vị
    width_percentiles = np.percentile(widths, [10, 25, 50, 75, 90, 95, 99])
    height_percentiles = np.percentile(heights, [10, 25, 50, 75, 90, 95, 99])
    area_percentiles = np.percentile(areas, [10, 25, 50, 75, 90, 95, 99])
    
    print("\nPhân vị chiều rộng (10, 25, 50, 75, 90, 95, 99):")
    print(width_percentiles)
    print("\nPhân vị chiều cao (10, 25, 50, 75, 90, 95, 99):")
    print(height_percentiles)
    print("\nPhân vị diện tích (10, 25, 50, 75, 90, 95, 99):")
    print(area_percentiles)
    
    # Kích thước khuôn mặt từ dataset (từ debug output)
    face_sizes = [
        # width, height, area
        (0.1562, 0.2984, 0.046631),
        (0.0672, 0.1063, 0.007139),
        (0.1047, 0.1891, 0.019792),
        (0.2688, 0.2422, 0.065088),
        (0.0047, 0.0109, 0.000066),
        (0.0406, 0.0609, 0.002476),
        (0.0047, 0.0078, 0.000037),
        (0.0406, 0.0625, 0.002539),
        (0.0094, 0.0141, 0.000146),
        (0.0266, 0.0391, 0.001038),
        (0.5125, 0.4547, 0.233027),
        (0.0047, 0.0094, 0.000059),
        (0.0359, 0.0781, 0.002686),
        (0.0047, 0.0156, 0.000088),
        (0.0109, 0.0344, 0.000376),
        (0.0906, 0.2094, 0.018975),
        (0.2812, 0.2609, 0.073389)
    ]
    
    # Trích xuất dữ liệu khuôn mặt
    face_widths = [f[0] for f in face_sizes]
    face_heights = [f[1] for f in face_sizes]
    face_areas = [f[2] for f in face_sizes]
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ phân bố chiều rộng anchor
    plt.subplot(2, 3, 1)
    plt.hist(widths, bins=50, alpha=0.7)
    plt.title('Chiều rộng anchor')
    plt.xlabel('Chiều rộng')
    plt.ylabel('Số lượng')
    
    # Biểu đồ phân bố chiều cao anchor
    plt.subplot(2, 3, 2)
    plt.hist(heights, bins=50, alpha=0.7)
    plt.title('Chiều cao anchor')
    plt.xlabel('Chiều cao')
    plt.ylabel('Số lượng')
    
    # Biểu đồ phân bố diện tích anchor
    plt.subplot(2, 3, 3)
    plt.hist(areas, bins=50, alpha=0.7)
    plt.title('Diện tích anchor')
    plt.xlabel('Diện tích')
    plt.ylabel('Số lượng')
    
    # So sánh chiều rộng
    plt.subplot(2, 3, 4)
    plt.boxplot([widths, face_widths], labels=['Anchor', 'Khuôn mặt'])
    plt.title('So sánh chiều rộng')
    plt.ylabel('Chiều rộng')
    
    # So sánh chiều cao
    plt.subplot(2, 3, 5)
    plt.boxplot([heights, face_heights], labels=['Anchor', 'Khuôn mặt'])
    plt.title('So sánh chiều cao')
    plt.ylabel('Chiều cao')
    
    # So sánh diện tích
    plt.subplot(2, 3, 6)
    plt.boxplot([areas, face_areas], labels=['Anchor', 'Khuôn mặt'])
    plt.title('So sánh diện tích')
    plt.ylabel('Diện tích')
    
    plt.tight_layout()
    plt.savefig('larger_anchors_comparison.png')
    print("Đã lưu biểu đồ so sánh vào larger_anchors_comparison.png")
    
    # Tính IoU giữa khuôn mặt và anchor phù hợp nhất
    print("\nTính IoU giữa khuôn mặt và anchor phù hợp nhất...")
    for i, (width, height, area) in enumerate(face_sizes):
        # Tạo hộp khuôn mặt [x1, y1, x2, y2] giả sử trung tâm tại (0.5, 0.5)
        face_box = np.array([
            0.5 - width/2, 0.5 - height/2, 0.5 + width/2, 0.5 + height/2
        ])
        
        # Tính IoU với tất cả anchor
        ious = calculate_ious(face_box, anchor_boxes_np)
        
        # Tìm anchor phù hợp nhất
        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]
        best_anchor = anchor_boxes_np[best_idx]
        best_width = best_anchor[2] - best_anchor[0]
        best_height = best_anchor[3] - best_anchor[1]
        best_area = best_width * best_height
        
        print(f"Khuôn mặt {i}: kích thước={width:.4f}×{height:.4f}, diện tích={area:.6f}")
        print(f"  Anchor phù hợp nhất: kích thước={best_width:.4f}×{best_height:.4f}, diện tích={best_area:.6f}, IoU={best_iou:.4f}")

def calculate_ious(box, boxes):
    """Tính IoU giữa một hộp và một mảng các hộp"""
    # Mở rộng hộp để khớp với shape của boxes
    box = box.reshape(1, 4)
    
    # Tính giao điểm
    xx1 = np.maximum(box[:, 0], boxes[:, 0])
    yy1 = np.maximum(box[:, 1], boxes[:, 1])
    xx2 = np.minimum(box[:, 2], boxes[:, 2])
    yy2 = np.minimum(box[:, 3], boxes[:, 3])
    
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    
    inter = w * h
    
    # Tính diện tích
    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Tính IoU
    union = area_box + area_boxes - inter
    iou = inter / union
    
    return iou.flatten()

if __name__ == "__main__":
    test_larger_anchors() 