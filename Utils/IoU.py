def calculate_centered_iou(
    box1: tuple[int | float, int | float],
    box2: tuple[int | float, int | float],
):
    width1, height1 = box1
    width2, height2 = box2

    area1 = width1 * height1
    area2 = width2 * height2

    intersection_width = min(width1, width2)
    intersection_height = min(height1, height2)
    intersection_area = intersection_width * intersection_height

    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area
