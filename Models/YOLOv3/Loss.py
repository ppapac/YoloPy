import torch.nn as nn
import torch

from Utils.IoU import calculate_centered_iou


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.l_class = 1
        self.l_nj = 10
        self.l_box = 10
        self.l_obj = 1

    def forward_step(
        self,
        predictions: list[torch.Tensor],
        target: list[torch.Tensor],
        anchors: list[int],
    ):
        obj = target[..., 0] == 1
        nj = target[..., 0] == 0

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][nj]),
            (target[..., 0:1][nj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        box_preds = torch.cat(
            [
                nn.Sigmoid(predictions[..., 1:3]),
                torch.exp(predictions[..., 3:5]) * anchors,
            ],
            dim=-1,
        )
        result = calculate_centered_iou(box_preds[obj], target[..., 1:5][obj]).detach()

        loss_obj = self.mse(
            nn.Sigmoid(predictions[..., 0:1][obj]), result * target[..., 0:1][obj]
        )

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid_function(predictions[..., 1:3])
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.cross_entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )

        return (
            self.l_box * box_loss
            + self.l_obj * loss_obj
            + self.l_nj * no_object_loss
            + self.l_class * class_loss
        )
