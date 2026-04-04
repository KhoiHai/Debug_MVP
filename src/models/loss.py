import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Build Targets
# ─────────────────────────────────────────────
def build_targets(outputs, targets, strides, num_classes):
    cls_targets = []
    box_targets = []
    obj_targets = []

    device = outputs["cls"][0].device

    for level, stride in enumerate(strides):
        B, _, H, W = outputs["cls"][level].shape

        cls_t = torch.zeros(B, num_classes, H, W, device=device)
        box_t = torch.zeros(B, 4, H, W, device=device)
        obj_t = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                gx = int(cx / stride)
                gy = int(cy / stride)

                if gx >= W or gy >= H:
                    continue

                tx = (cx / stride) - gx
                ty = (cy / stride) - gy
                tw = torch.log(w / stride + 1e-6)
                th = torch.log(h / stride + 1e-6)

                obj_t[b, 0, gy, gx] = 1.0
                cls_t[b, label, gy, gx] = 1.0

                # ✅ FIX: dùng stack (KHÔNG dùng torch.tensor)
                box_t[b, :, gy, gx] = torch.stack([tx, ty, tw, th])

        cls_targets.append(cls_t)
        box_targets.append(box_t)
        obj_targets.append(obj_t)

    return cls_targets, box_targets, obj_targets


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
class Model_Loss(nn.Module):
    def __init__(self, num_classes=20, strides=[8,16,32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

    def forward(self, outputs, targets):
        cls_preds = outputs["cls"]
        box_preds = outputs["box"]
        obj_preds = outputs["obj"]

        cls_t, box_t, obj_t = build_targets(
            outputs, targets, self.strides, self.num_classes
        )

        loss_cls = 0.0
        loss_box = 0.0
        loss_obj = 0.0

        for i in range(len(cls_preds)):
            pred_cls = cls_preds[i]   # [B,C,H,W]
            pred_box = box_preds[i]   # [B,4,H,W]
            pred_obj = obj_preds[i]   # [B,1,H,W]

            target_cls = cls_t[i]
            target_box = box_t[i]
            target_obj = obj_t[i]

            # ─────────────────────────
            # OBJECTNESS LOSS (toàn grid)
            # ─────────────────────────
            loss_obj += F.binary_cross_entropy_with_logits(
                pred_obj,
                target_obj,
                reduction='mean'   # ổn định hơn
            )

            # ─────────────────────────
            # POSITIVE MASK
            # ─────────────────────────
            pos_mask = target_obj.squeeze(1) == 1   # [B,H,W]

            num_pos = pos_mask.sum().clamp(min=1)

            if pos_mask.sum() > 0:
                # ───────── CLS ─────────
                pred_cls_pos = pred_cls.permute(0,2,3,1)[pos_mask]   # [N_pos, C]
                target_cls_pos = target_cls.permute(0,2,3,1)[pos_mask]

                loss_cls += F.binary_cross_entropy_with_logits(
                    pred_cls_pos,
                    target_cls_pos,
                    reduction='sum'
                ) / num_pos

                # ───────── BOX ─────────
                pred_box_pos = pred_box.permute(0,2,3,1)[pos_mask]   # [N_pos, 4]
                target_box_pos = target_box.permute(0,2,3,1)[pos_mask]

                loss_box += F.smooth_l1_loss(
                    pred_box_pos,
                    target_box_pos,
                    reduction='sum'
                ) / num_pos

        total_loss = loss_cls + loss_box + loss_obj

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_obj": loss_obj
        }