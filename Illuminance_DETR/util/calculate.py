import torch
import numpy as np

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=1)

def box_cxcywhd_to_xyzxyz(x):
    cx, cy, cz, w, h, d = x.unbind(-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cz - 0.5 * d,
         cx + 0.5 * w, cy + 0.5 * h, cz + 0.5 * d]
    return torch.stack(b, dim=-1)

def box_xyzxyz_to_cxcywhd(x):
    x1, y1, z1, x2, y2, z2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
         (x2 - x1), (y2 - y1), (z2 - z1)]
    return torch.stack(b, dim=-1)

def box_iou_3d(boxes1, boxes2):
    vol1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    vol2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    inter_xyz1 = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    inter_xyz2 = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])

    inter_whd = (inter_xyz2 - inter_xyz1).clamp(min=0)
    
    intersection_volume = inter_whd[:, :, 0] * inter_whd[:, :, 1] * inter_whd[:, :, 2]

    union_volume = vol1[:, None] + vol2 - intersection_volume

    iou = intersection_volume / (union_volume + 1e-6)
    return iou, union_volume

def calculate_ap_3d(all_preds, all_gts, num_classes, iou_thresholds):
    preds_by_class = {c: [] for c in range(num_classes)}
    
    for i, pred in enumerate(all_preds):
        image_id = all_gts[i]['image_id']
        for score, label, box in zip(pred['scores'], pred['labels'], pred['boxes']):
            c = label.item()
            if c < num_classes:
                preds_by_class[c].append({
                    'image_id': image_id,
                    'score': score.item(),
                    'box': box
                })

    ap_matrix = np.zeros((num_classes, len(iou_thresholds)))
    class_exists = np.zeros(num_classes, dtype=bool)

    for i_thresh, iou_thresh in enumerate(iou_thresholds):
        gts_by_class = {c: {} for c in range(num_classes)}
        for gt in all_gts:
            image_id = gt['image_id']
            for label, box in zip(gt['labels'], gt['boxes']):
                c = label.item()
                if c >= num_classes: continue
                if image_id not in gts_by_class[c]:
                    gts_by_class[c][image_id] = {'boxes': [], 'matched': []}
                gts_by_class[c][image_id]['boxes'].append(box)
                gts_by_class[c][image_id]['matched'].append(False)

        for c in range(num_classes):
            preds = preds_by_class[c]
            gts = gts_by_class[c]

            total_gts = sum(len(img_gts['boxes']) for img_gts in gts.values())
            
            if total_gts == 0:
                continue
            
            class_exists[c] = True

            if len(preds) == 0:
                ap_matrix[c, i_thresh] = 0.0
                continue

            preds.sort(key=lambda x: x['score'], reverse=True)

            tps = np.zeros(len(preds))
            fps = np.zeros(len(preds))

            for i, pred in enumerate(preds):
                image_id = pred['image_id']
                
                if image_id not in gts or len(gts[image_id]['boxes']) == 0:
                    fps[i] = 1.0
                    continue

                gt_boxes_tensor = torch.stack(gts[image_id]['boxes']).to(pred['box'].device)
                pred_box_tensor = pred['box'].unsqueeze(0)

                gt_xyz = box_cxcywhd_to_xyzxyz(gt_boxes_tensor)
                pred_xyz = box_cxcywhd_to_xyzxyz(pred_box_tensor)
                
                ious, _ = box_iou_3d(pred_xyz, gt_xyz)
                ious = ious[0]

                best_gt_idx = torch.argmax(ious).item()
                best_iou = ious[best_gt_idx].item()
                
                if best_iou >= iou_thresh and not gts[image_id]['matched'][best_gt_idx]:
                    tps[i] = 1.0
                    gts[image_id]['matched'][best_gt_idx] = True
                else:
                    fps[i] = 1.0

            tp_cumsum = np.cumsum(tps)
            fp_cumsum = np.cumsum(fps)
            
            recalls = tp_cumsum / (total_gts + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            ap = 0.0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11.0
            
            ap_matrix[c, i_thresh] = ap

    class_aps = {}
    for c in range(num_classes):
        if class_exists[c]:
            class_aps[c] = np.mean(ap_matrix[c, :])
        else:
            class_aps[c] = 0.0

    if np.sum(class_exists) > 0:
        mAP = np.mean(ap_matrix[class_exists, :])
    else:
        mAP = 0.0

    return mAP, class_aps