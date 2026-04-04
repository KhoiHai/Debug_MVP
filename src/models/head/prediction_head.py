import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neck.panet_neck import ConvNormAct

class Prediction_Head(nn.Module):
    def __init__(self, in_channels=256, num_classes=80, num_prototypes=32, num_convs=2,
                 norm="bn", act="silu"):
        super().__init__()

        shared = []
        for _ in range(num_convs):
            shared.append(ConvNormAct(in_channels, in_channels, 3, p=1, norm=norm, act=act))
        self.shared_conv = nn.Sequential(*shared)

        # YOLO branches
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, 4, 1)
        self.obj_head = nn.Conv2d(in_channels, 1, 1) 

        # (sau này segmentation)
        # self.coef_head = nn.Conv2d(in_channels, num_prototypes, 1)

    def forward(self, features):
        cls_outputs = []
        box_outputs = []
        obj_outputs = []

        for f in features:
            x = self.shared_conv(f)

            cls = self.cls_head(x)
            box = self.box_head(x)     
            obj = self.obj_head(x)

            cls_outputs.append(cls)
            box_outputs.append(box)
            obj_outputs.append(obj)

        return cls_outputs, box_outputs, obj_outputs