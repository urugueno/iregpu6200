"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

import util.misc as utils
from util.box_ops import box_iou, box_cxcywh_to_xyxy

def box_cxcywhd_to_xyzxyz(x):
    """ (cx,cy,cz,w,h,d) -> (x1,y1,z1,x2,y2,z2) """
    cx, cy, cz, w, h, d = x.unbind(-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cz - 0.5 * d,
         cx + 0.5 * w, cy + 0.5 * h, cz + 0.5 * d]
    return torch.stack(b, dim=-1)

def box_iou_3d(boxes1, boxes2):
    """
    3D Bounding BoxのIoU (Intersection over Union) をペアワイズで計算する。
    boxes1: (N, 6), format (x1, y1, z1, x2, y2, z2)
    boxes2: (M, 6), format (x1, y1, z1, x2, y2, z2)
    Returns: iou (N, M), union_volume (N, M)
    """
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
    """
    3D BBox用のAverage Precision (AP) を計算する。
    指定されたすべてのIoU閾値で計算を行い、その平均を返す。
    
    Returns:
        mAP (float): 全クラス・全閾値での平均AP
        class_aps (dict): {class_id: ap} 各クラスの平均AP
    """
    
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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('iou', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
    if 'loss_center' in criterion.weight_dict:
        metric_logger.add_meter('CNN_error', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1))
    if 'loss_center' in criterion.weight_dict:
        metric_logger.add_meter('loss_center', utils.SmoothedValue(window_size=1))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = {k: v.to(device) for k, v in samples.items()}
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        if 'selected_sensor_indices' in outputs:
            indices = outputs['selected_sensor_indices'][0].detach().cpu().numpy()
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if 'loss' in loss_dict_reduced_scaled:
            metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss'])
        if 'loss_ce' in loss_dict_reduced_scaled:
            metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss_ce'])
        if 'loss_bbox' in loss_dict_reduced_scaled:
            metric_logger.update(loss_bbox=loss_dict_reduced_scaled['loss_bbox'])
        if 'loss_giou' in loss_dict_reduced_scaled:
            metric_logger.update(loss_giou=loss_dict_reduced_scaled['loss_giou'])
        if 'loss_center' in loss_dict_reduced_scaled:
            metric_logger.update(loss_center=loss_dict_reduced_scaled['loss_center'])
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        with torch.no_grad():
            indices = criterion.matcher(outputs, targets)
            batch_acc = []
            batch_iou = []
            
            total_tp = 0
            total_fn = 0

            for i, (pred_idx, tgt_idx) in enumerate(indices):
                predicted_logits = outputs['pred_logits'][i, pred_idx]
                predicted_labels = predicted_logits.argmax(-1)
                
                target_labels = targets[i]['labels'][tgt_idx]
                acc = (predicted_labels == target_labels).float().mean()
                batch_acc.append(acc)

                predicted_boxes_6d = outputs['pred_boxes'][i, pred_idx]
                target_boxes_6d = targets[i]['boxes'][tgt_idx]

                iou, _ = box_iou_3d(
                    box_cxcywhd_to_xyzxyz(predicted_boxes_6d), 
                    box_cxcywhd_to_xyzxyz(target_boxes_6d)
                )
                iou = iou.diag().mean()
                batch_iou.append(iou)

                total_tp += (predicted_labels == target_labels).sum().item()
                total_fn += len(targets[i]['labels']) - len(tgt_idx)


            if batch_acc:
                 metric_logger.update(accuracy=torch.stack(batch_acc).mean())
            if batch_iou:
                 metric_logger.update(iou=torch.stack(batch_iou).mean())
            
            pred_logits = outputs['pred_logits']
            background_class_idx = pred_logits.shape[-1] - 1
            num_predictions_as_object = (pred_logits.argmax(-1) != background_class_idx).sum().item()
            total_fp = num_predictions_as_object - total_tp
            
            epsilon = 1e-6
            precision = total_tp / (total_tp + total_fp + epsilon)
            recall = total_tp / (total_tp + total_fn + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
            if 'pred_center_coords' in outputs and outputs['pred_center_coords'] is not None:
                valid_targets = [t for t in targets if 'center_coords' in t]
                if valid_targets:
                    pred_coords = outputs['pred_center_coords']
                    batch_indices = [i for i, t in enumerate(targets) if 'center_coords' in t]
                    pred_coords_filtered = pred_coords[batch_indices]
                    true_coords = torch.stack([t['center_coords'] for t in valid_targets])
                    distances = torch.norm(pred_coords_filtered - true_coords, p=2, dim=1)
                    avg_dist_error = distances.mean().item()
                    metric_logger.update(CNN_error=avg_dist_error)
            
            if not np.isnan(f1_score):
                metric_logger.update(f1_score=f1_score)
        
    metric_logger.synchronize_between_processes()
    print("\n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, header_prefix: str = 'Test', prediction_filename: str = None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('iou', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
    if 'loss_center' in criterion.weight_dict:
        metric_logger.add_meter('CNN_error', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1))
    if 'loss_center' in criterion.weight_dict:
        metric_logger.add_meter('loss_center', utils.SmoothedValue(window_size=1))
    header = f'{header_prefix}:'

    predictions_for_csv = []
    all_preds_for_ap = []
    all_gts_for_ap = []
    coord_predictions_for_csv = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = {k: v.to(device) for k, v in samples.items()}
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)

        selected_sensor_ids_batch = None
        selected_sensor_coords_batch = None
        
        if 'selected_sensor_indices' in outputs and outputs['selected_sensor_indices'] is not None:
            selected_sensor_ids_batch = outputs['selected_sensor_indices'].cpu().numpy()
            
            if 'coords' in samples:
                all_coords = samples['coords'].cpu().numpy()
                selected_sensor_coords_batch = []
                for i in range(len(selected_sensor_ids_batch)):
                    indices = selected_sensor_ids_batch[i]
                    coords = all_coords[i][indices]
                    selected_sensor_coords_batch.append(coords)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        if 'loss' in loss_dict_reduced_scaled:
            metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss'])
        if 'loss_ce' in loss_dict_reduced_scaled:
            metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss_ce'])
        if 'loss_bbox' in loss_dict_reduced_scaled:
            metric_logger.update(loss_bbox=loss_dict_reduced_scaled['loss_bbox'])
        if 'loss_giou' in loss_dict_reduced_scaled:
            metric_logger.update(loss_giou=loss_dict_reduced_scaled['loss_giou'])
        if 'loss_center' in loss_dict_reduced_scaled:
            metric_logger.update(loss_center=loss_dict_reduced_scaled['loss_center'])
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        indices = criterion.matcher(outputs, targets)

        batch_acc = []
        batch_iou = []
        total_tp = 0
        total_fn = 0
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            predicted_logits = outputs['pred_logits'][i, pred_idx]
            predicted_labels = predicted_logits.argmax(-1)
            target_labels = targets[i]['labels'][tgt_idx]

            acc = (predicted_labels == target_labels).float().mean()
            batch_acc.append(acc)

            predicted_boxes_6d = outputs['pred_boxes'][i, pred_idx]
            target_boxes_6d = targets[i]['boxes'][tgt_idx]

            iou, _ = box_iou_3d(
                box_cxcywhd_to_xyzxyz(predicted_boxes_6d), 
                box_cxcywhd_to_xyzxyz(target_boxes_6d)
            )
            iou = iou.diag().mean()
            batch_iou.append(iou)
            
            total_tp += (predicted_labels == target_labels).sum().item()
            total_fn += len(targets[i]['labels']) - len(tgt_idx)

        pred_logits = outputs['pred_logits']
        background_class_idx = pred_logits.shape[-1] - 1
        num_predictions_as_object = (pred_logits.argmax(-1) != background_class_idx).sum().item()
        total_fp = num_predictions_as_object - total_tp
        
        epsilon = 1e-6
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        if 'pred_center_coords' in outputs and outputs['pred_center_coords'] is not None:
            valid_targets = [t for t in targets if 'center_coords' in t]
            if valid_targets:
                pred_coords = outputs['pred_center_coords']
                batch_indices = [i for i, t in enumerate(targets) if 'center_coords' in t]
                pred_coords_filtered = pred_coords[batch_indices]
                true_coords = torch.stack([t['center_coords'] for t in valid_targets])
                distances = torch.norm(pred_coords_filtered - true_coords, p=2, dim=1)
                avg_dist_error = distances.mean().item()
                metric_logger.update(CNN_error=avg_dist_error)
        
        if batch_acc:
             metric_logger.update(accuracy=torch.stack(batch_acc).mean())
        if batch_iou:
             metric_logger.update(iou=torch.stack(batch_iou).mean())
        if not np.isnan(f1_score):
             metric_logger.update(f1_score=f1_score)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)


        out_logits = outputs['pred_logits']
        out_bbox = outputs['pred_boxes']
        prob = F.softmax(out_logits, -1)
        
        scores, labels = prob.max(-1)

        keep = labels != criterion.num_classes
        
        
        for i in range(len(targets)):
            all_preds_for_ap.append({
                'scores': scores[i][keep[i]].cpu(),
                'labels': labels[i][keep[i]].cpu(),
                'boxes': out_bbox[i][keep[i]].cpu() 
            })
            
            all_gts_for_ap.append({
                'labels': targets[i]['labels'].cpu(),
                'boxes': targets[i]['boxes'].cpu(),
                'image_id': targets[i]['image_id'].item()
            })

        if 'pred_center_coords' in outputs and outputs['pred_center_coords'] is not None:
            pred_coords_cpu = outputs['pred_center_coords'].cpu().numpy()
            for i, target in enumerate(targets):
                if 'center_coords' in target:
                    true_coords_cpu = target['center_coords'].cpu().numpy()
                    coord_predictions_for_csv.append({
                        'image_path': target.get('image_path', 'N/A'),
                        'true_cx': true_coords_cpu[0],
                        'true_cy': true_coords_cpu[1],
                        'pred_cx': pred_coords_cpu[i][0],
                        'pred_cy': pred_coords_cpu[i][1],
                    })
        for i, res in enumerate(results):
            pred_boxes = res['boxes']
            pred_scores = res['scores']
            pred_labels = res['labels']

            image_path = targets[i]['image_path']
            true_label = targets[i]['labels'][0].item()

            sensor_ids_str = ""
            sensor_coords_str = ""
            
            if selected_sensor_ids_batch is not None:
                ids = selected_sensor_ids_batch[i]
                sensor_ids_str = str(list(ids))
                
                if selected_sensor_coords_batch is not None:
                    coords = selected_sensor_coords_batch[i]
                    coords_list = [f"({c[0]:.2f}, {c[1]:.2f})" for c in coords]
                    sensor_coords_str = str(coords_list).replace("'", "")

            true_box_cxcywhd_norm = targets[i]['boxes']
            true_box_xyzxyz_norm = box_cxcywhd_to_xyzxyz(true_box_cxcywhd_norm)
            
            h, w, d = targets[i]['orig_size']
            scale_fct = torch.tensor([w, h, d, w, h, d], device=device)
            
            true_box_xyzxyz = (true_box_xyzxyz_norm * scale_fct)[0].cpu().numpy()
            true_x1, true_y1, true_z1, true_x2, true_y2, true_z2 = true_box_xyzxyz

            if len(pred_boxes) == 0:
                pred_dict = {
                    'image_path': image_path,
                    'pred_label': np.nan,
                    'pred_x1': np.nan, 'pred_y1': np.nan, 'pred_z1': np.nan,
                    'pred_x2': np.nan, 'pred_y2': np.nan, 'pred_z2': np.nan,
                    'score': np.nan,
                    'true_label': true_label,
                    'true_x1': true_x1, 'true_y1': true_y1, 'true_z1': true_z1,
                    'true_x2': true_x2, 'true_y2': true_y2, 'true_z2': true_z2,
                }
                if sensor_coords_str:
                    pred_dict['selected_sensor_coords'] = sensor_coords_str
                predictions_for_csv.append(pred_dict)
            else:
                for box, score, p_label in zip(pred_boxes.cpu().numpy(), pred_scores.cpu().numpy(), pred_labels.cpu().numpy()):
                    pred_x1, pred_y1, pred_z1, pred_x2, pred_y2, pred_z2 = box
                    pred_dict = {
                        'image_path': image_path,
                        'pred_label': p_label.item(),
                        'pred_x1': pred_x1, 'pred_y1': pred_y1, 'pred_z1': pred_z1,
                        'pred_x2': pred_x2, 'pred_y2': pred_y2, 'pred_z2': pred_z2,
                        'score': score,
                        'true_label': true_label,
                        'true_x1': true_x1, 'true_y1': true_y1, 'true_z1': true_z1,
                        'true_x2': true_x2, 'true_y2': true_y2, 'true_z2': true_z2,
                    }
                    if sensor_coords_str:
                        pred_dict['selected_sensor_coords'] = sensor_coords_str
                    predictions_for_csv.append(pred_dict)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

    metric_logger.synchronize_between_processes()
    print("\n")

    if header_prefix == 'Test' and utils.is_main_process() and len(all_preds_for_ap) > 0:
        print("\n" + "="*60)
        print("--- 3D AP Evaluation Results ---")
        
        mAP50, class_ap50 = calculate_ap_3d(
            all_preds_for_ap, all_gts_for_ap, criterion.num_classes, [0.50]
        )
        
        mAP75, class_ap75 = calculate_ap_3d(
            all_preds_for_ap, all_gts_for_ap, criterion.num_classes, [0.75]
        )

        ap_thresholds_avg = np.arange(0.5, 1.0, 0.05)
        mAP, class_ap = calculate_ap_3d(
            all_preds_for_ap, all_gts_for_ap, criterion.num_classes, ap_thresholds_avg
        )

        print(f"\nOverall Performance:")
        print(f"  mAP    (0.50:0.95): {mAP:.4f}")
        print(f"  mAP50  (0.50)     : {mAP50:.4f}")
        print(f"  mAP75  (0.75)     : {mAP75:.4f}")

        print(f"\nPer-Class Performance:")
        print(f"  {'Class ID':<10} | {'AP (0.5:0.95)':<15} | {'AP50':<10} | {'AP75':<10}")
        print("-" * 55)
        
        for c in range(criterion.num_classes):
            ap_val = class_ap.get(c, 0.0)
            ap50_val = class_ap50.get(c, 0.0)
            ap75_val = class_ap75.get(c, 0.0)
            print(f"  {c:<10} | {ap_val:<15.4f} | {ap50_val:<10.4f} | {ap75_val:<10.4f}")
        
        print("="*60 + "\n")

        metric_logger.update(ap=mAP, ap50=mAP50, ap75=mAP75)
        for c in range(criterion.num_classes):
            metric_logger.update(**{
                f'AP_class_{c}': class_ap.get(c, 0.0),
                f'AP50_class_{c}': class_ap50.get(c, 0.0),
                f'AP75_class_{c}': class_ap75.get(c, 0.0)
            })
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    df_preds = pd.DataFrame(predictions_for_csv)
    df_coords = pd.DataFrame(coord_predictions_for_csv)

    if output_dir and prediction_filename:
        df_preds.to_csv(os.path.join(output_dir, prediction_filename), index=False)
        print(f"Predictions saved to {os.path.join(output_dir, prediction_filename)}\n")

    return stats, _, df_preds, df_coords