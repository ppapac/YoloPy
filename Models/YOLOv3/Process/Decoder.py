import torch

IMAGE_HEIGHT = 420
IMAGE_WIDTH = 760
ANCHORS = [13, 26, 52]


def decode_raw_output(network_output_box: torch.Tensor, i: int):
    """Transform raw network output to real dimension."""

    output_shape = network_output_box.size()  # [batch_size, 51, width, height]
    batch_size = output_shape[0]
    output_width = output_shape[2]
    output_height = output_shape[3]

    network_output_box = torch.stack(
        torch.chunk(network_output_box, 3, dim=1),
        dim=1,
    )  # [batch_size, 3, 17, width, height]

    raw_output_offsets = network_output_box[:, :, :2, :, :]
    raw_widths_heights = network_output_box[:, :, 2:4, :, :]
    raw_confidences = network_output_box[:, :, 4:5, :, :]
    raw_category_probabilities = network_output_box[:, :, 5:, :, :]

    x = torch.arange(0, output_height, 1)
    y = torch.arange(0, output_width, 1)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)
    xy_grid = torch.stack((x_grid, y_grid), dim=2)
    xy_grid = xy_grid.repeat(batch_size, 3, 1, 1, 1)

    # Calculate the center position of the prediction box:
    predict_real_xy = torch.sigmoid(raw_output_offsets) + xy_grid
    predict_real_xy[:, :, 0, :, :] *= IMAGE_HEIGHT / output_height
    predict_real_xy[:, :, 1, :, :] *= IMAGE_WIDTH / output_width
    # Calculate the length and width of the prediction box:
    predict_real_wh = torch.exp(raw_widths_heights) * ANCHORS[i]
    predict_real_wh[:, :, 0, :, :] *= IMAGE_HEIGHT / output_height
    predict_real_wh[:, :, 1, :, :] *= IMAGE_WIDTH / output_width

    predict_real_bbox = torch.cat((predict_real_xy, predict_real_wh), axis=2)
    predict_confidences = torch.sigmoid(
        raw_confidences
    )  # object box calculates the predicted confidence
    predict_probabilities = torch.sigmoid(
        raw_category_probabilities
    )  # calculating the predicted probability category box object
    return torch.concat(
        [predict_real_bbox, predict_confidences, predict_probabilities], axis=2
    )
