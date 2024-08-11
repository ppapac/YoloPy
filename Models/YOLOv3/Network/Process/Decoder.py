

import numpy as np
import torch


def cell_to_bbox(network_output: list[torch.Tensor], NUM_CLASS, i=0):
    """Transform raw network output to real dimension."""
    # where i = 0, 1 or 2 to correspond to the three grid scales  
    output_shape = torch.Size(network_output)
    batch_size = output_shape[0]
    output_width = output_shape[2]
    output_height = output_shape[3]

    network_output = torch.stack(torch.chunk(network_output, 3, dim=1))

    raw_output_offsets = network_output[:, :, :2, :, :]     
    raw_width_height = network_output[:, :, 2:4, :, :]
    confidcene = network_output[:, :, 4:5, :, :]
    category_probabilities = network_output[:, :, 5:, :, :]

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    y = torch.arange(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

 
