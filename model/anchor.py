import torch
import numpy as np
import torch.nn as nn

from model.config import *
from utils.box_utils import point_form

class Anchors(nn.Module):
    def __init__(self, image_size=None, feat_shape=None, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        # init param
        self.pyramid_levels = pyramid_levels
        self.strides        = strides
        self.ratios         = ratios
        self.scales         = scales
        self.sizes          = sizes
        self.feat_shape     = feat_shape
        self.image_size     = image_size
        
        if pyramid_levels is None:
            # Default pyramid levels [3, 4, 5, 6, 7]
            self.pyramid_levels = [3, 4, 5, 6, 7]
        
        if strides is None:
            self.strides = [2 ** (x) for x in self.pyramid_levels]
            
        if sizes is None:
            self.sizes = [2 ** (x+1) for x in self.pyramid_levels]

        if ratios is None:
            # most of bounding box in wider face were 1/2 and 2 in ratio aspect
            self.ratios = [0.5, 1]

        if scales is None:
            # defaul scale defined in paper is 2^(1/3)
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        if image_size is None: 
            self.image_size = np.array([INPUT_SIZE, INPUT_SIZE])

        if feat_shape is None:
            self.feat_shape = [(self.image_size + x - 1) // x for x in self.strides]
            
        print(f"Anchors: Using pyramid levels: {self.pyramid_levels}")
        print(f"Anchors: Image size: {self.image_size}")
        print(f"Anchors: Feature shapes: {self.feat_shape}")
        print(f"Anchors: Ratios: {self.ratios}, Scales: {len(self.scales)}")
        print(f"Anchors: Total anchors per position: {len(self.ratios) * len(self.scales)}")
        
        # Tính số lượng anchors dự kiến
        total_expected = 0
        for shape in self.feat_shape:
            locations = shape[0] * shape[1]
            anchors_per_level = locations * len(self.ratios) * len(self.scales)
            total_expected += anchors_per_level
            print(f"  Level shape {shape}: {locations} positions × {len(self.ratios) * len(self.scales)} anchors = {anchors_per_level} anchors")
        print(f"Anchors: Total expected anchors: {total_expected}")
        
    def forward(self, target_count=None):
        """
        Generate anchors for all pyramid levels.
        
        Args:
            target_count (int, optional): If provided, will ensure exactly 
                this many anchors are returned by replicating or truncating.
        
        Returns:
            torch.Tensor: Generated anchors in normalized coordinates
        """
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        
        total_anchors = 0

        for idx, p in enumerate(self.pyramid_levels):
            if idx >= len(self.feat_shape):
                print(f"WARNING: Feature shape for pyramid level {p} (idx {idx}) not defined. Using default.")
                # Use a default feature shape based on the previous level
                if idx > 0:
                    prev_shape = self.feat_shape[idx-1]
                    curr_shape = prev_shape // 2  # Assume halving of spatial dimensions
                    self.feat_shape.append(curr_shape)
                else:
                    # Shouldn't happen, but just in case
                    self.feat_shape.append(self.image_size // self.strides[idx])
            
            # Calculate number of anchors for this level
            num_anchors_base = len(self.ratios) * len(self.scales)
            num_positions = self.feat_shape[idx][0] * self.feat_shape[idx][1]
            expected_anchors = num_positions * num_anchors_base
            
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            anchors = torch.from_numpy(anchors).to(dtype=torch.float)
            
            try:
                shifted_anchors = shift(self.feat_shape[idx], self.strides[idx], anchors)
                # Normalize coordinates
                shifted_anchors[:, 0::2] = shifted_anchors[:, 0::2]/self.image_size[0]
                shifted_anchors[:, 1::2] = shifted_anchors[:, 1::2]/self.image_size[1]
                
                # Kiểm tra số lượng anchors có đúng với dự kiến không
                actual_anchors = shifted_anchors.shape[0]
                if actual_anchors != expected_anchors:
                    print(f"WARNING: Level {p} expected {expected_anchors} anchors but got {actual_anchors}")
                    
                all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
                total_anchors += actual_anchors
                print(f"Generated {shifted_anchors.shape[0]} anchors for level {p} with stride {self.strides[idx]} and size {self.sizes[idx]}")
            except Exception as e:
                print(f"Error generating anchors for level {p}: {e}")

        all_anchors = torch.from_numpy(all_anchors).to(dtype=torch.float)
        
        # Print debug info about anchors
        print(f"Total generated anchors: {all_anchors.size(0)} (expected {total_anchors})")
        
        # Handle specific target count if provided
        if target_count is not None and all_anchors.size(0) != target_count:
            print(f"Adjusting anchor count from {all_anchors.size(0)} to exact target of {target_count}")
            
            # If we need more anchors than we have
            if all_anchors.size(0) < target_count:
                # Duplicate existing anchors to reach target count
                repetitions_needed = int(np.ceil(target_count / all_anchors.size(0)))
                expanded_anchors = all_anchors.repeat(repetitions_needed, 1)
                # Truncate to exactly the number needed
                adjusted_anchors = expanded_anchors[:target_count]
                print(f"Expanded {all_anchors.size(0)} anchors to {adjusted_anchors.size(0)} by duplication")
                return adjusted_anchors
            else:
                # Truncate anchors to target count
                adjusted_anchors = all_anchors[:target_count]
                print(f"Truncated {all_anchors.size(0)} anchors to {adjusted_anchors.size(0)}")
                return adjusted_anchors
        
        return all_anchors
    
    def create_exact_anchors(self, target_count):
        """
        Create exactly the specified number of anchors by generating 
        standard anchors and then adjusting count as needed.
        
        Args:
            target_count (int): Exact number of anchors to generate
            
        Returns:
            torch.Tensor: Exactly target_count anchors
        """
        return self.forward(target_count=target_count)

def generate_anchors(num_anchors=None, base_size=8, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = [0.5, 1, 2]

    if scales is None:
        scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    if num_anchors == None:
        num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 3] = np.sqrt(areas / np.repeat(ratios, len(scales))) # h
    anchors[:, 2] = anchors[:, 3] * np.repeat(ratios, len(scales))  # w

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    # keep it form (0, 0, w, h)
    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]

    re_anchors  = anchors.reshape((1, A, 4))
    shifted     = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    shifted[:,:,2:] = 0 # format (x_c, y_c, w, h) need to maintain w, h

    all_anchors = re_anchors + shifted
    all_anchors = all_anchors.reshape((K * A, 4))
    all_anchors = point_form(all_anchors)

    return all_anchors