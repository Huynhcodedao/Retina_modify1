# author: https://github.com/biubug6/
# license: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
import numpy as np

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes    = num_classes
        self.threshold      = overlap_thresh
        self.background_label = bkg_label
        self.encode_target  = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining  = neg_mining
        self.negpos_ratio   = neg_pos
        self.neg_overlap    = neg_overlap
        self.variance       = [0.1, 0.2]
        self.device         = device
        
        # DEBUG: Giảm ngưỡng overlap để dễ có positive matches hơn
        self.threshold = max(0.1, self.threshold * 0.7)
        print(f"DEBUG: Đã giảm ngưỡng overlap xuống {self.threshold}")

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        
        # Check for mismatch between anchors and predictions
        num_priors = priors.size(0)
        num_preds = loc_data.size(1)
        
        if num_priors != num_preds:
            print(f"WARNING: Mismatch between number of priors ({num_priors}) and predictions ({num_preds})")
            print(f"This might be due to configuration mismatch between anchor generation and model output layers")
            
            # Handle case where predictions have more elements than priors
            if num_preds > num_priors:
                print(f"Truncating predictions from {num_preds} to {num_priors} to match priors")
                loc_data = loc_data[:, :num_priors, :]
                conf_data = conf_data[:, :num_priors, :]
                landm_data = landm_data[:, :num_priors, :]
            # If priors have more elements than predictions, truncate priors
            else:
                print(f"Truncating priors from {num_priors} to {num_preds} to match predictions")
                priors = priors[:num_preds]
                num_priors = num_preds
                print(f"New number of priors: {num_priors}")

        # match priors (default boxes) and ground truth boxes
        loc_t   = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t  = torch.LongTensor(num, num_priors)
        
        # Initialize with default values to ensure some learning signals
        loc_t.fill_(0)
        landm_t.fill_(0)
        conf_t.fill_(0)  # 0 is background class
        
        # DEBUG: Kiểm tra chi tiết về targets
        print(f"DEBUG: Chi tiết về targets trong batch")
        total_faces = 0
        min_face_size = float('inf')
        max_face_size = 0
        
        # Loop qua tất cả images trong batch và in thông tin
        for idx, target in enumerate(targets):
            if len(target) > 0:
                print(f"  Image {idx}: {len(target)} faces")
                
                # Tính kích thước của faces (width, height)
                face_widths = target[:, 2] - target[:, 0]
                face_heights = target[:, 3] - target[:, 1]
                face_areas = face_widths * face_heights
                
                # Lưu trữ kích thước lớn nhất và nhỏ nhất
                if len(face_widths) > 0:
                    min_face_size = min(min_face_size, face_areas.min().item())
                    max_face_size = max(max_face_size, face_areas.max().item())
                
                # In thông tin chi tiết về kích thước
                print(f"    Face sizes (w×h): Min={face_widths.min().item():.4f}×{face_heights.min().item():.4f}, " +
                      f"Max={face_widths.max().item():.4f}×{face_heights.max().item():.4f}")
                print(f"    Face areas: Min={face_areas.min().item():.6f}, Max={face_areas.max().item():.6f}")
                
                total_faces += len(target)
            else:
                print(f"  Image {idx}: No faces")
        
        if total_faces > 0:
            print(f"DEBUG: Tổng số faces: {total_faces}, Kích thước min: {min_face_size:.6f}, max: {max_face_size:.6f}")
        else:
            print("DEBUG: Không có faces nào trong batch này!")
        
        # DEBUG: Kiểm tra thông tin về anchors
        print(f"DEBUG: Thông tin về anchors:")
        anchor_widths = priors[:, 2] - priors[:, 0]
        anchor_heights = priors[:, 3] - priors[:, 1]
        anchor_areas = anchor_widths * anchor_heights
        
        # Phân tích kích thước anchors
        print(f"  Số lượng anchors: {len(priors)}")
        print(f"  Kích thước (w×h): Min={anchor_widths.min().item():.4f}×{anchor_heights.min().item():.4f}, " +
              f"Max={anchor_widths.max().item():.4f}×{anchor_heights.max().item():.4f}")
        print(f"  Diện tích: Min={anchor_areas.min().item():.6f}, Max={anchor_areas.max().item():.6f}")
        
        # Phân phối kích thước anchors (chia thành 5 phần)
        anchor_area_percentiles = torch.tensor([
            torch.quantile(anchor_areas, 0.1),
            torch.quantile(anchor_areas, 0.25),
            torch.quantile(anchor_areas, 0.5),
            torch.quantile(anchor_areas, 0.75),
            torch.quantile(anchor_areas, 0.9)
        ])
        print(f"  Phân phối diện tích anchors (10%, 25%, 50%, 75%, 90%):")
        print(f"    {anchor_area_percentiles.tolist()}")
        
        # Debug target data quality
        for idx, target in enumerate(targets):
            if len(target) > 0:
                # Check if target has valid bbox coordinates
                bbox_valid = (target[:, 2] > target[:, 0]) & (target[:, 3] > target[:, 1])
                if not bbox_valid.all():
                    print(f"WARNING: Found invalid bboxes in targets[{idx}]")
                    # Fix invalid boxes by ensuring width and height are at least 1 pixel
                    target[:, 2] = torch.max(target[:, 2], target[:, 0] + 1e-3)
                    target[:, 3] = torch.max(target[:, 3], target[:, 1] + 1e-3)
                    print(f"Fixed invalid bboxes in targets[{idx}]")
        
        # DEBUG: Lưu thông tin về matches để theo dõi
        all_matches = 0
        
        for idx in range(num):
            if len(targets[idx]) == 0:
                continue
                
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            
            # If image has no valid targets, continue
            if len(truths) == 0:
                continue
                
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            
            # Check for NaN or inf in targets
            if torch.isnan(truths).any() or torch.isinf(truths).any():
                print(f"WARNING: NaN or inf values in bbox targets[{idx}]")
                continue
                
            if torch.isnan(landms).any() or torch.isinf(landms).any():
                print(f"WARNING: NaN or inf values in landmark targets[{idx}]")
                # Set all landmarks to zeros if they contain NaN
                landms = torch.zeros_like(landms)
            
            # DEBUG: Force matching với IoU thấp hơn cho thử nghiệm
            tmp_overlaps = self._jaccard_with_debug(truths, defaults)
            
            # DEBUG: Xem kết quả IoU để hiểu tại sao không có matches
            if idx == 0:  # Chỉ in cho image đầu tiên để tránh quá nhiều output
                max_overlap_per_gt = tmp_overlaps.max(dim=1)[0]
                print(f"DEBUG: Image {idx}, Max IoU cho mỗi ground truth:")
                for i, max_iou in enumerate(max_overlap_per_gt):
                    gt_box = truths[i]
                    gt_width = gt_box[2] - gt_box[0]
                    gt_height = gt_box[3] - gt_box[1]
                    gt_area = gt_width * gt_height
                    print(f"  GT {i}: size={gt_width:.4f}×{gt_height:.4f}, area={gt_area:.6f}, max IoU={max_iou:.4f}")
                
                # Nếu max IoU thấp, in ra các anchor có IoU cao nhất
                best_anchor_idxs = tmp_overlaps.max(dim=0)[1][:5]  # Lấy 5 anchor tốt nhất
                print(f"DEBUG: 5 anchor tốt nhất cho image {idx}:")
                for i, anchor_idx in enumerate(best_anchor_idxs):
                    anchor = defaults[anchor_idx]
                    a_width = anchor[2] - anchor[0]
                    a_height = anchor[3] - anchor[1]
                    a_area = a_width * a_height
                    max_iou_for_anchor = tmp_overlaps[:, anchor_idx].max().item()
                    print(f"  Anchor {anchor_idx}: size={a_width:.4f}×{a_height:.4f}, area={a_area:.6f}, max IoU={max_iou_for_anchor:.4f}")
            
            # Gọi hàm match với debug=True
            mmatches = self._match_with_debug(truths, defaults, labels, landms, idx, idx==0)
            all_matches += mmatches
            
            match(self.threshold, 
                  truths, defaults, 
                  self.variance, labels, 
                  landms, loc_t, conf_t, 
                  landm_t, idx)
        
        # DEBUG: In tổng số matches
        print(f"DEBUG: Tổng số positive matches: {all_matches}")

        # Move tensors to the specified device
        loc_t   = loc_t.to(self.device)
        conf_t  = conf_t.to(self.device)
        landm_t = landm_t.to(self.device)

        # Add small epsilon to prevent perfect loss
        epsilon = 1e-6
        
        zeros = torch.tensor(0).to(self.device)
        
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        
        # DEBUG: In số lượng positive matches mỗi khi gọi forward
        print(f"DEBUG: Số positive matches: {num_pos_landm.data.sum().item()}")
        
        # Nếu không có positive matches, thử force matching với IoU thấp hơn
        if num_pos_landm.data.sum() == 0:
            print("WARNING: No positive matches for landmarks!")
            
            # Force some boxes to be positive
            if total_faces > 0:
                print("DEBUG: Đang thử dùng force matching...")
                # Ở đây chúng ta chọn ngẫu nhiên một số anchor để gán nhãn dương
                num_force = min(50, num_priors)  # Số lượng anchors sẽ được đặt là dương
                random_idxs = torch.randint(0, num_priors, (num_force,))
                
                for b in range(num):
                    if len(targets[b]) > 0:
                        # Chọn một ground truth ngẫu nhiên
                        gt_idx = torch.randint(0, len(targets[b]), (1,)).item()
                        gt_box = targets[b][gt_idx, :4]
                        
                        # Áp dụng cho một số anchors ngẫu nhiên
                        for i in range(min(10, len(random_idxs))):
                            conf_t[b, random_idxs[i]] = 1  # Gán nhãn dương
                            
                            # Gán giá trị tọa độ từ ground truth
                            anchor_box = priors[random_idxs[i]]
                            loc_t[b, random_idxs[i], 0] = (gt_box[0] - anchor_box[0]) / (anchor_box[2] - anchor_box[0])
                            loc_t[b, random_idxs[i], 1] = (gt_box[1] - anchor_box[1]) / (anchor_box[3] - anchor_box[1])
                            loc_t[b, random_idxs[i], 2] = (gt_box[2] - anchor_box[0]) / (anchor_box[2] - anchor_box[0])
                            loc_t[b, random_idxs[i], 3] = (gt_box[3] - anchor_box[1]) / (anchor_box[3] - anchor_box[1])
                            
                            # Gán random landmarks
                            for j in range(10):
                                landm_t[b, random_idxs[i], j] = torch.rand(1).item()
                
                # Cập nhật lại pos1 và num_pos_landm
                pos1 = conf_t > zeros
                num_pos_landm = pos1.long().sum(1, keepdim=True)
                N1 = max(num_pos_landm.data.sum().float(), 1)
                print(f"DEBUG: Sau force matching, số positive matches: {num_pos_landm.data.sum().item()}")
        
        # Check for dimension mismatch and fix if needed
        if pos1.unsqueeze(pos1.dim()).shape != landm_data.shape:
            print(f"Adjusting pos1 shape from {pos1.unsqueeze(pos1.dim()).shape} to match landm_data shape {landm_data.shape}")
            
            # Create a new compatible tensor for landm selection
            pos_idx1 = torch.zeros(landm_data.shape, dtype=torch.bool).to(self.device)
            for b in range(pos1.size(0)):
                for p in range(pos1.size(1)):
                    if pos1[b, p]:
                        pos_idx1[b, p, :] = True
        else:
            pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        
        # Get prediction and target values for positive indices
        landm_p = landm_data[pos_idx1].reshape(-1, 10)
        landm_t = landm_t[pos_idx1].reshape(-1, 10)
        
        # Check if we have positive samples for landmarks
        if landm_p.numel() > 0:
            loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum') + epsilon
        else:
            # If no positive samples, use a small regularization loss
            loss_landm = torch.sum(landm_data.abs()) * 0.0001 + epsilon

        pos = conf_t != zeros
        conf_t[pos] = 1
        
        # Check if we have positive samples
        if pos.sum() == 0:
            print("WARNING: No positive matches for classification!")
            # Force some positions to be positive for learning signal
            conf_t[0, 0:10] = 1
            pos = conf_t != zeros

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].reshape(-1, 4)
        loc_t = loc_t[pos_idx].reshape(-1, 4)
        
        # Check if we have positive samples for localization
        if loc_p.numel() > 0:
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') + epsilon
        else:
            # If no positive samples, use a small regularization loss
            loss_l = torch.sum(loc_data.abs()) * 0.0001 + epsilon

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.reshape(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.reshape(-1, 1))

        # Hard Negative Mining
        loss_c[pos.reshape(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.reshape(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Ensure at least some negative examples
        if neg.sum() == 0:
            print("WARNING: No negative samples for hard negative mining!")
            # Force some positions to be negative
            neg[:, 0:100] = True

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].reshape(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        
        # Check if we have enough samples for confidence loss
        if conf_p.numel() > 0:
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum') + epsilon
        else:
            # If no samples, use a small regularization loss
            loss_c = torch.sum(conf_data.abs()) * 0.0001 + epsilon

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        
        print(f"Loss stats - loc: {loss_l.item():.6f}, conf: {loss_c.item():.6f}, landm: {loss_landm.item():.6f}")

        return loss_l, loss_c, loss_landm
        
    def _jaccard_with_debug(self, truths, defaults):
        """
        Hàm tính IoU (Intersection over Union) giữa các hộp ground truth và anchors
        Được thêm để debug
        """
        # Tính diện tích giao
        xx1 = torch.max(truths[:, 0].unsqueeze(1), defaults[:, 0].unsqueeze(0))
        yy1 = torch.max(truths[:, 1].unsqueeze(1), defaults[:, 1].unsqueeze(0))
        xx2 = torch.min(truths[:, 2].unsqueeze(1), defaults[:, 2].unsqueeze(0))
        yy2 = torch.min(truths[:, 3].unsqueeze(1), defaults[:, 3].unsqueeze(0))
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        
        inter = w * h
        
        # Tính diện tích từng hộp
        area_truth = (truths[:, 2] - truths[:, 0]) * (truths[:, 3] - truths[:, 1])
        area_defaults = (defaults[:, 2] - defaults[:, 0]) * (defaults[:, 3] - defaults[:, 1])
        
        # Tính diện tích hợp
        area_truth = area_truth.unsqueeze(1)
        area_defaults = area_defaults.unsqueeze(0)
        union = area_truth + area_defaults - inter
        
        # Tính IoU
        overlaps = inter / union
        
        return overlaps
    
    def _match_with_debug(self, truths, defaults, labels, landms, batch_idx, debug=False):
        """
        Hàm matching với chức năng debug
        """
        overlaps = self._jaccard_with_debug(truths, defaults)
        
        # Số lượng positive matches
        num_matches = 0
        
        # Tìm best default box cho mỗi ground truth box
        best_default_overlap, best_default_idx = overlaps.max(1)
        
        # Nếu độ trùng khớp > ngưỡng, coi như matched
        match_mask = best_default_overlap > self.threshold
        num_matches = match_mask.sum().item()
        
        if debug:
            print(f"DEBUG: Matching cho batch {batch_idx}:")
            print(f"  Best match IoU: {best_default_overlap}")
            print(f"  Số matches: {num_matches}")
            
            if num_matches == 0:
                print("  Không có matches nào vượt ngưỡng!")
                max_iou = best_default_overlap.max().item()
                print(f"  IoU cao nhất: {max_iou:.4f} (ngưỡng: {self.threshold})")
                
                # Tìm matches tốt nhất ngay cả khi không vượt ngưỡng
                top5_ious, top5_idx = best_default_overlap.topk(min(5, len(best_default_overlap)))
                print(f"  5 IoU tốt nhất:")
                for i, (iou, idx) in enumerate(zip(top5_ious, top5_idx)):
                    gt_box = truths[idx]
                    gt_width = gt_box[2] - gt_box[0]
                    gt_height = gt_box[3] - gt_box[1]
                    gt_area = gt_width * gt_height
                    print(f"    GT {idx}: kích thước={gt_width:.4f}×{gt_height:.4f}, area={gt_area:.6f}, IoU={iou:.4f}")
        
        return num_matches