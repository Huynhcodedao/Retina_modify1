import torch
import numpy as np
import matplotlib.pyplot as plt
from model.anchor import Anchors

def visualize_anchors():
    """
    Visualize the anchor sizes and compare with face sizes from the dataset
    """
    print("Generating anchors with new parameters...")
    
    # Create anchors with our new parameters
    anchors = Anchors()
    anchor_boxes = anchors.forward()
    
    # Convert to numpy for easier manipulation
    anchor_boxes_np = anchor_boxes.numpy()
    
    # Calculate widths and heights
    widths = anchor_boxes_np[:, 2] - anchor_boxes_np[:, 0]
    heights = anchor_boxes_np[:, 3] - anchor_boxes_np[:, 1]
    areas = widths * heights
    
    # Print statistics
    print(f"Total anchors: {len(anchor_boxes_np)}")
    print(f"Width range: {widths.min():.6f} to {widths.max():.6f}")
    print(f"Height range: {heights.min():.6f} to {heights.max():.6f}")
    print(f"Area range: {areas.min():.6f} to {areas.max():.6f}")
    
    # Calculate percentiles
    width_percentiles = np.percentile(widths, [10, 25, 50, 75, 90])
    height_percentiles = np.percentile(heights, [10, 25, 50, 75, 90])
    area_percentiles = np.percentile(areas, [10, 25, 50, 75, 90])
    
    print("\nWidth percentiles (10, 25, 50, 75, 90):")
    print(width_percentiles)
    print("\nHeight percentiles (10, 25, 50, 75, 90):")
    print(height_percentiles)
    print("\nArea percentiles (10, 25, 50, 75, 90):")
    print(area_percentiles)
    
    # Sample face sizes from the dataset (from debug output)
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
        (0.0109, 0.0344, 0.000376)
    ]
    
    # Extract face data
    face_widths = [f[0] for f in face_sizes]
    face_heights = [f[1] for f in face_sizes]
    face_areas = [f[2] for f in face_sizes]
    
    # Plot histograms of anchor and face dimensions
    plt.figure(figsize=(15, 10))
    
    # Plot anchor widths
    plt.subplot(2, 3, 1)
    plt.hist(widths, bins=50, alpha=0.7)
    plt.title('Anchor Widths')
    plt.xlabel('Width')
    plt.ylabel('Count')
    
    # Plot anchor heights
    plt.subplot(2, 3, 2)
    plt.hist(heights, bins=50, alpha=0.7)
    plt.title('Anchor Heights')
    plt.xlabel('Height')
    plt.ylabel('Count')
    
    # Plot anchor areas
    plt.subplot(2, 3, 3)
    plt.hist(areas, bins=50, alpha=0.7)
    plt.title('Anchor Areas')
    plt.xlabel('Area')
    plt.ylabel('Count')
    
    # Plot comparison of widths
    plt.subplot(2, 3, 4)
    plt.boxplot([widths, face_widths], labels=['Anchors', 'Faces'])
    plt.title('Width Comparison')
    plt.ylabel('Width')
    
    # Plot comparison of heights
    plt.subplot(2, 3, 5)
    plt.boxplot([heights, face_heights], labels=['Anchors', 'Faces'])
    plt.title('Height Comparison')
    plt.ylabel('Height')
    
    # Plot comparison of areas
    plt.subplot(2, 3, 6)
    plt.boxplot([areas, face_areas], labels=['Anchors', 'Faces'])
    plt.title('Area Comparison')
    plt.ylabel('Area')
    
    plt.tight_layout()
    plt.savefig('anchor_visualization.png')
    print("Saved visualization to anchor_visualization.png")
    
    # Calculate IoU between faces and best matching anchors
    print("\nCalculating IoU between faces and anchors...")
    for i, (width, height, area) in enumerate(face_sizes):
        # Create face box [x1, y1, x2, y2] assuming center at (0.5, 0.5)
        face_box = np.array([
            0.5 - width/2, 0.5 - height/2, 0.5 + width/2, 0.5 + height/2
        ])
        
        # Calculate IoU with all anchors
        ious = calculate_ious(face_box, anchor_boxes_np)
        
        # Find best matching anchor
        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]
        best_anchor = anchor_boxes_np[best_idx]
        best_width = best_anchor[2] - best_anchor[0]
        best_height = best_anchor[3] - best_anchor[1]
        best_area = best_width * best_height
        
        print(f"Face {i}: size={width:.4f}×{height:.4f}, area={area:.6f}")
        print(f"  Best anchor: size={best_width:.4f}×{best_height:.4f}, area={best_area:.6f}, IoU={best_iou:.4f}")

def calculate_ious(box, boxes):
    """Calculate IoU between one box and an array of boxes"""
    # Expand box to match shape of boxes
    box = box.reshape(1, 4)
    
    # Calculate intersection
    xx1 = np.maximum(box[:, 0], boxes[:, 0])
    yy1 = np.maximum(box[:, 1], boxes[:, 1])
    xx2 = np.minimum(box[:, 2], boxes[:, 2])
    yy2 = np.minimum(box[:, 3], boxes[:, 3])
    
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    
    inter = w * h
    
    # Calculate areas
    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Calculate IoU
    union = area_box + area_boxes - inter
    iou = inter / union
    
    return iou.flatten()

if __name__ == "__main__":
    visualize_anchors() 