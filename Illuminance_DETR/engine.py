import math
import os
import sys
from typing import Iterable

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

import util.misc as utils
import util.append
from util.calculate import box_cxcywhd_to_xyzxyz, box_iou_3d, calculate_ap_3d

def train(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    
    task = args.task
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    if task == 'pose':
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('mpjpe', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_x', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_y', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_z', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_keypoints', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))
    else:
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('iou', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))

    header = 'Epoch: [{}]'.format(epoch)

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        if isinstance(samples, dict):
            samples = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        else:
            samples = samples.to(device)
            
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if task == 'pose':
            metric_logger.update(loss=loss_value)
            if 'loss_keypoint' in loss_dict_scaled:
                metric_logger.update(loss_keypoints=loss_dict_scaled['loss_keypoint'])
        else:
            metric_logger.update(loss=loss_value)
            if 'loss_bbox' in loss_dict_scaled:
                metric_logger.update(loss_bbox=loss_dict_scaled['loss_bbox'])
        if 'loss' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss'])
        if 'loss_ce' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss_ce'])
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])

        with torch.no_grad():
            indices = criterion.matcher(outputs, targets)

            batch_acc = []
            total_tp = 0
            total_fn = 0

            if task == 'pose':
                batch_mpjpe = []
                batch_mpjdle_x = []
                batch_mpjdle_y = []
                batch_mpjdle_z = []
            else:
                batch_iou = []

            for i, (pred_idx, tgt_idx) in enumerate(indices):
                if len(tgt_idx) == 0:
                    continue

                predicted_labels = outputs['pred_logits'][i, pred_idx].argmax(-1)
                target_labels = targets[i]['labels'][tgt_idx]
                acc = (predicted_labels == target_labels).float().mean()
                batch_acc.append(acc)

                if task == 'pose':
                    predicted_kpts = outputs['pred_keypoints'][i, pred_idx].reshape(-1, 14, 3)
                    target_kpts = targets[i]['keypoints'][tgt_idx]

                    scale_fct = torch.tensor(args.real_space_size, device=device)
                    scale_fct = scale_fct.view(1, 1, 3)

                    diff = (predicted_kpts - target_kpts) * scale_fct
                    dist = torch.norm(diff, dim=2)
                    
                    if dist.numel() > 0:
                        batch_mpjpe.append(dist.mean())
                        
                    diff_xyz = diff.abs()
                    batch_mpjdle_x.append(diff_xyz[..., 0].mean())
                    batch_mpjdle_y.append(diff_xyz[..., 1].mean())
                    batch_mpjdle_z.append(diff_xyz[..., 2].mean())
                    
                else:
                    predicted_boxes_6d = outputs['pred_boxes'][i, pred_idx]
                    target_boxes_6d = targets[i]['boxes'][tgt_idx]

                    iou_matrix, _ = box_iou_3d(
                        box_cxcywhd_to_xyzxyz(predicted_boxes_6d), 
                        box_cxcywhd_to_xyzxyz(target_boxes_6d)
                    )
                    iou = iou_matrix.diag().mean()
                    batch_iou.append(iou)

                total_tp += (predicted_labels == target_labels).sum().item()
                total_fn += len(targets[i]['labels']) - len(tgt_idx)

            if batch_acc:
                metric_logger.update(accuracy=torch.stack(batch_acc).mean())
            
            if task == 'pose':
                if batch_mpjpe:
                    metric_logger.update(mpjpe=torch.stack(batch_mpjpe).mean())
                if batch_mpjdle_x:
                    metric_logger.update(mpjdle_x=torch.stack(batch_mpjdle_x).mean())
                    metric_logger.update(mpjdle_y=torch.stack(batch_mpjdle_y).mean())
                    metric_logger.update(mpjdle_z=torch.stack(batch_mpjdle_z).mean())
            else:
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
            
            if not np.isnan(f1_score):
                metric_logger.update(f1_score=f1_score)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, 
             header_prefix: str = 'Test', prediction_filename: str = None, base_ds=None, args=None):
    model.eval()
    criterion.eval()

    task = args.task
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    if task == 'pose':
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('mpjpe', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_x', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_y', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_z', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_keypoints', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))
    else:
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('iou', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))

    header = f'{header_prefix}:'

    predictions_for_csv = []
    all_preds_for_ap = []
    all_gts_for_ap = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        if isinstance(samples, dict):
            samples = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        else:
            samples = samples.to(device)
            
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
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}

        if task == 'pose':
            if 'loss_keypoint' in loss_dict_scaled:
                metric_logger.update(loss_keypoints=loss_dict_scaled['loss_keypoint'])
        else:
            if 'loss_bbox' in loss_dict_scaled:
                metric_logger.update(loss_bbox=loss_dict_scaled['loss_bbox'])
        if 'loss' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss'])
        if 'loss_ce' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss_ce'])
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])
        indices = criterion.matcher(outputs, targets)

        batch_acc = []
        total_tp = 0
        total_fn = 0

        if task == 'pose':
            batch_mpjpe, batch_mpjdle_x, batch_mpjdle_y, batch_mpjdle_z = [], [], [], []
        else:
            batch_iou = []

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue

            predicted_labels = outputs['pred_logits'][i, pred_idx].argmax(-1)
            target_labels = targets[i]['labels'][tgt_idx]
            acc = (predicted_labels == target_labels).float().mean()
            batch_acc.append(acc)

            if task == 'pose':
                predicted_kpts = outputs['pred_keypoints'][i, pred_idx].reshape(-1, 14, 3)
                target_kpts = targets[i]['keypoints'][tgt_idx]
            
                scale_fct = torch.tensor(args.real_space_size, device=device)
                scale_fct = scale_fct.view(1, 1, 3)
            
                diff = (predicted_kpts - target_kpts) * scale_fct
                dist = torch.norm(diff, dim=2)
            
                if dist.numel() > 0:
                    batch_mpjpe.append(dist.mean())
                
                diff_xyz = diff.abs()
                batch_mpjdle_x.append(diff_xyz[..., 0].mean())
                batch_mpjdle_y.append(diff_xyz[..., 1].mean())
                batch_mpjdle_z.append(diff_xyz[..., 2].mean())

            else:
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

        if batch_acc: metric_logger.update(accuracy=torch.stack(batch_acc).mean())
        if task == 'pose':
            if batch_mpjpe: metric_logger.update(mpjpe=torch.stack(batch_mpjpe).mean())
            if batch_mpjdle_x:
                metric_logger.update(mpjdle_x=torch.stack(batch_mpjdle_x).mean())
                metric_logger.update(mpjdle_y=torch.stack(batch_mpjdle_y).mean())
                metric_logger.update(mpjdle_z=torch.stack(batch_mpjdle_z).mean())
        else:
            if batch_iou: metric_logger.update(iou=torch.stack(batch_iou).mean())

        pred_logits = outputs['pred_logits']
        background_class_idx = pred_logits.shape[-1] - 1
        num_predictions_as_object = (pred_logits.argmax(-1) != background_class_idx).sum().item()
        total_fp = num_predictions_as_object - total_tp
        epsilon = 1e-6
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        if not np.isnan(f1): metric_logger.update(f1_score=f1)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if task == 'pose':
            for i, res in enumerate(results):
                sensor_coords_str = ""
                
                if selected_sensor_coords_batch is not None:
                    coords = selected_sensor_coords_batch[i]
                    coords_list = [f"({c[0]:.2f}, {c[1]:.2f})" for c in coords]
                    sensor_coords_str = str(coords_list).replace("'", "")

                util.append.append_pose_predictions(
                    predictions_for_csv, res, targets[i], sensor_coords_str
                )
        else:
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

            for i, res in enumerate(results):
                sensor_coords_str = ""
                
                if selected_sensor_ids_batch is not None:
                    if selected_sensor_coords_batch is not None:
                        coords = selected_sensor_coords_batch[i]
                        coords_list = [f"({c[0]:.2f}, {c[1]:.2f})" for c in coords]
                        sensor_coords_str = str(coords_list).replace("'", "")

                util.append.append_bbox_predictions(
                    predictions_for_csv, res, targets[i], device, sensor_coords_str
                )

    metric_logger.synchronize_between_processes()
    print("")

    if len(predictions_for_csv) > 0:
        if task == 'pose' and header_prefix == 'Test':
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print(f"--- Final Evaluation Summary ---")
            print(f"\nOverall Pose Estimation Performance:")
            
            acc = stats.get('accuracy', float('nan'))
            f1 = stats.get('f1_score', float('nan'))
            mpjpe = stats.get('mpjpe', float('nan'))
            mpjdle_x = stats.get('mpjdle_x', float('nan'))
            mpjdle_y = stats.get('mpjdle_y', float('nan'))
            mpjdle_z = stats.get('mpjdle_z', float('nan'))

            print(f"MPJPE     : {mpjpe:.4f}")
            print(f"MPJDLE X  : {mpjdle_x:.4f}")
            print(f"MPJDLE Y  : {mpjdle_y:.4f}")
            print(f"MPJDLE Z  : {mpjdle_z:.4f}")
            print(f"Accuracy  : {acc:.4f}")
            print(f"F1 Score  : {f1:.4f}")
                
            print("-"*55 + "\n")

        elif task == 'bbox' and header_prefix == 'Test':
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

            print(f"--- Final Evaluation Summary ---")
            print(f"\nOverall Object Detection Performance:")
            print(f"  mAP    (0.50:0.95): {mAP:.4f}")
            print(f"  mAP50  (0.50)     : {mAP50:.4f}")
            print(f"  mAP75  (0.75)     : {mAP75:.4f}")

            print(f"\nPer-Class Performance:")
            print(f"  {'Class ID':<10} | {'AP (0.5:0.95)':<15} | {'AP50':<10} | {'AP75':<10}")
            print("-" * 55)
            for c in range(criterion.num_classes):
                print(f"  {c:<10} | {class_ap.get(c, 0.0):<15.4f} | {class_ap50.get(c, 0.0):<10.4f} | {class_ap75.get(c, 0.0):<10.4f}")

            metric_logger.update(ap=mAP, ap50=mAP50, ap75=mAP75)
            for c in range(criterion.num_classes):
                metric_logger.update(**{
                    f'AP_class_{c}': class_ap.get(c, 0.0),
                    f'AP50_class_{c}': class_ap50.get(c, 0.0),
                    f'AP75_class_{c}': class_ap75.get(c, 0.0)
                })
            print("")

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if prediction_filename and output_dir:
            csv_path = os.path.join(output_dir, prediction_filename)
            pd.DataFrame(predictions_for_csv).to_csv(csv_path, index=False)

    return stats