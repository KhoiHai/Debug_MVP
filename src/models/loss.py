import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.match_locations import match_locations
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations


# ═════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═════════════════════════════════════════════════════════════
def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    logits: [N, C]
    targets: [N] (>=0: class, -1: negative)
    """
    N, C = logits.shape

    targets_onehot = torch.zeros_like(logits)
    pos_mask = targets >= 0
    targets_onehot[pos_mask, targets[pos_mask].long()] = 1.0

    prob = torch.sigmoid(logits)

    ce = F.binary_cross_entropy_with_logits(
        logits, targets_onehot, reduction='none'
    )

    p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
    focal_weight = (1 - p_t) ** gamma

    alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)

    loss = ce * focal_weight * alpha_t

    return loss.sum()


# ═════════════════════════════════════════════════════════════
# CLASSIFICATION ONLY LOSS
# ═════════════════════════════════════════════════════════════
class Model_Loss(nn.Module):
    def __init__(
        self,
        num_classes=20,
        alpha_cls=1.0,
        strides=[8, 16, 32],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_cls = alpha_cls
        self.strides = strides

    def forward(self, outputs, targets):
        """
        outputs: { "cls": list of [B, C, Hi, Wi] }
        targets: list of dicts {boxes, labels}
        """

        # ═════════════════════════════════════════
        # STEP 1: Flatten
        # ═════════════════════════════════════════
        cls_preds = flatten_predictions(outputs["cls"])  # [B, N, C]
        B, N, C = cls_preds.shape
        total_predictions = B * N

        # ═════════════════════════════════════════
        # STEP 2: Locations
        # ═════════════════════════════════════════
        locations = generate_locations(outputs["cls"], self.strides)
        locations = locations.to(cls_preds.device)

        all_cls_loss = 0.0

        # ═════════════════════════════════════════
        # STEP 3: Loop từng image
        # ═════════════════════════════════════════
        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(cls_preds.device)
            gt_labels = targets[i]["labels"].to(cls_preds.device)

            # ❌ Không có object → toàn negative
            if gt_boxes.shape[0] == 0:
                gt_cls_target = torch.full(
                    (N,), -1, dtype=torch.long, device=cls_preds.device
                )
                all_cls_loss += sigmoid_focal_loss(cls_preds[i], gt_cls_target)
                continue

            # ═════════════════════════════════════
            # MATCH LOCATIONS
            # ═════════════════════════════════════
            matched_idx, pos_mask = match_locations(locations, gt_boxes)

            # tạo target
            gt_cls_target = torch.full(
                (N,), -1, dtype=torch.long, device=cls_preds.device
            )

            if pos_mask.sum() > 0:
                gt_cls_target[pos_mask] = gt_labels[matched_idx[pos_mask]]

            # ═════════════════════════════════════
            # NEGATIVE MINING (1:3)
            # ═════════════════════════════════════
            neg_mask = ~pos_mask

            num_pos = pos_mask.sum()
            num_neg = neg_mask.sum()

            max_neg = 100 if num_pos == 0 else num_pos * 3

            if num_neg > max_neg:
                neg_idx = torch.where(neg_mask)[0]
                perm = torch.randperm(num_neg, device=neg_idx.device)[:max_neg]
                selected_neg = neg_idx[perm]

                new_neg_mask = torch.zeros_like(neg_mask)
                new_neg_mask[selected_neg] = True
                neg_mask = new_neg_mask

            final_mask = pos_mask | neg_mask

            cls_pred_sampled = cls_preds[i][final_mask]
            gt_cls_sampled = gt_cls_target[final_mask]

            all_cls_loss += sigmoid_focal_loss(
                cls_pred_sampled, gt_cls_sampled
            )

        # ═════════════════════════════════════════
        # NORMALIZE
        # ═════════════════════════════════════════
        final_loss_cls = (all_cls_loss / total_predictions) * self.alpha_cls

        return {
            "loss_cls": final_loss_cls
        }