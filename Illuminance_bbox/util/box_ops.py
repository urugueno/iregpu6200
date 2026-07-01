"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    if x.shape[-1] == 4:
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
    elif x.shape[-1] == 6:
        x_c, y_c, z_c, w, h, d = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
             (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    else:
        raise ValueError(f"Unsupported box dimension: {x.shape[-1]}")
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    if x.shape[-1] == 4:
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
    elif x.shape[-1] == 6:
        x0, y0, z0, x1, y1, z1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
             (x1 - x0), (y1 - y0), (z1 - z0)]
    else:
        raise ValueError(f"Unsupported box dimension: {x.shape[-1]}")
    return torch.stack(b, dim=-1)

def box_volume(boxes):
    """
    Computes the volume of a set of bounding boxes (x1, y1, z1, x2, y2, z2).
    For 2D boxes, it returns area.
    """
    if boxes.shape[-1] == 4:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    elif boxes.shape[-1] == 6:
        return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])
    else:
        raise ValueError(f"Unsupported box dimension: {boxes.shape[-1]}")

def box_iou(boxes1, boxes2):
    vol1 = box_volume(boxes1)
    vol2 = box_volume(boxes2)

    if boxes1.shape[-1] == 4:
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
    else:
        lt = torch.max(boxes1[:, None, :3], boxes2[:, :3]) 
        rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])
        whd = (rb - lt).clamp(min=0)
        inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]
    union = vol1[:, None] + vol2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    dim = boxes1.shape[-1]
    if dim == 4:
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    elif dim == 6:
        assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
        assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    if dim == 4:
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        area = wh[:, :, 0] * wh[:, :, 1]
    else:
        lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
        rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])
        whd = (rb - lt).clamp(min=0)
        area = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
