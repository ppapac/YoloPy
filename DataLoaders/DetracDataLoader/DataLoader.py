import pathlib
import os

import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

from Utils.IoU import calculate_centered_iou


class DetracDataLoader(Dataset):
    def __init__(
        self,
        annotations_folder: pathlib.Path,
        image_folder: pathlib.Path,
        anchors: list[float | int],
        feature_output_shapes: list[tuple[int]],
        transform: callable | None = None,
        target_transform: callable | None = None,
    ):
        self.annotations_folder = annotations_folder
        self.image_folder = image_folder
        self.transform = transform
        self.target_transform = target_transform
        self.anchors = anchors
        self.feature_output_shapes = feature_output_shapes

    def __len__(self):
        return len(os.listdir(self.image_folder))

    def __getitem__(self, idx: int):
        images = os.listdir(self.image_folder)
        img_path = self.img_dir / images[idx]
        image = read_image(img_path)
        annotations_path = self.annotations_folder / images[idx]
        annotations_path = annotations_path.parent / (annotations_path.stem + ".xml")
        annotations = parse_objects_from_xml(annotations_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            annotations = self.target_transform(annotations)

        targets_by_scale = [
            make_target_tensor(
                annotations,
                self.anchors,
                feature_output_width,
                feature_scale_height,
                0.1,
            )
            for feature_output_width, feature_scale_height in self.feature_output_shapes
        ]
        ground_truth = torch.cat(targets_by_scale)

        return image, annotations


def parse_objects_from_xml(
    annotations_path: pathlib.Path,
) -> list[dict[str, float | str]]:
    annotations = ET.parse(annotations_path)
    annotations_root = annotations.getroot()
    bndboxes = []
    for obj in annotations_root.findall("object"):
        name = obj.find("name").text
        color = obj.find("color").text
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        bndboxes.append(
            {
                "name": name,
                "color": color,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    return bndboxes


def transform_annotation_xyxy_to_xywh(
    box: dict[str, float | str]
) -> dict[str, float | str]:
    return {
        "name": box["name"],
        "color": box["color"],
        "x": (box["xmin"] + box["xmax"]) / 2,
        "y": (box["ymin"] + box["ymax"]) / 2,
        "w": (box["xmax"] - box["xmin"]),
        "h": (box["ymax"] - box["ymin"]),
    }


def make_target_tensor(
    annotations: list[dict],
    anchors: list[float],
    feature_scale_width: int,
    feature_scale_height: int,
    ignore_anchor_threshold: float,
) -> torch.Tensor:
    anchors_scaled = [
        (anchor * feature_scale_width, anchor * feature_scale_height)
        for anchor in anchors
    ]
    target = torch.zeros((len(anchors), 6, feature_scale_width, feature_scale_height))
    for box in annotations:
        box_assigned_anchor = False
        box_xywh = transform_annotation_xyxy_to_xywh(box)
        ious_with_anchors = [
            calculate_centered_iou((box_xywh["w"], box_xywh["h"]), anchor)
            for anchor in anchors_scaled
        ]
        anchor_indices_sorted_by_iou = np.argsort(ious_with_anchors)[::-1]

        for anchor_index in anchor_indices_sorted_by_iou:

            i, j = int(feature_scale_height * box_xywh["x"]), int(
                feature_scale_width * box_xywh["y"]
            )
            anchor_taken = target[anchor_index, 0, i, j]
            if not anchor_taken and not box_assigned_anchor:
                target[anchor_index, 0, i, j] = 1
                target[anchor_index, 1:5, i, j] = [
                    feature_scale_height * box_xywh["x"] - i,
                    feature_scale_height * box_xywh["y"] - i,
                    box_xywh["w"] * feature_scale_width,
                    box_xywh["h"] * feature_scale_height,
                ]
                box_assigned_anchor = True
            elif (
                not anchor_taken
                and box_assigned_anchor
                and ious_with_anchors[anchor_index] > ignore_anchor_threshold
            ):
                target[anchor_index, 0, i, j] = -1
    return torch.Tensor(target)
