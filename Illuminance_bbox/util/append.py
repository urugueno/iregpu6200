import numpy as np
import torch
from .calculate import box_cxcywhd_to_xyzxyz

def append_pose_predictions(predictions_for_csv, res, target, sensor_coords_str):
    kpt_names = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        "NECK"
    ]
    image_path = target['image_path']
    pred_kpts = res['keypoints'].cpu().numpy()
    pred_scores = res['scores'].cpu().numpy()
    pred_labels = res['labels'].cpu().numpy()

    gt_kpts_norm = target['keypoints'].cpu()
    img_h, img_w, img_d = target['orig_size'].tolist()
    scale = torch.tensor([img_w, img_h, img_d]).view(1, 1, 3)
    gt_kpts = (gt_kpts_norm * scale).numpy()

    if len(pred_scores) == 0:
        row = {
            'image_path': image_path,
            'score': np.nan,
            'pred_label': np.nan,
        }
        for name in kpt_names:
            row[f'pred_{name}_X'] = np.nan
            row[f'pred_{name}_Y'] = np.nan
            row[f'pred_{name}_Z'] = np.nan
        
        if len(gt_kpts) > 0:
            g_kpt = gt_kpts[0]
            for k_idx, name in enumerate(kpt_names):
                row[f'gt_{name}_X'] = g_kpt[k_idx, 0]
                row[f'gt_{name}_Y'] = g_kpt[k_idx, 1]
                row[f'gt_{name}_Z'] = g_kpt[k_idx, 2]
        if sensor_coords_str:
            row['selected_sensor_coords'] = sensor_coords_str
        
        predictions_for_csv.append(row)
    
    else:
        for j in range(len(pred_scores)):
            row = {
                'image_path': image_path,
                'score': pred_scores[j],
                'pred_label': pred_labels[j],
            }
            
            p_kpt = pred_kpts[j]
            for k_idx, name in enumerate(kpt_names):
                row[f'pred_{name}_X'] = p_kpt[k_idx, 0]
                row[f'pred_{name}_Y'] = p_kpt[k_idx, 1]
                row[f'pred_{name}_Z'] = p_kpt[k_idx, 2]

            if len(gt_kpts) > 0:
                g_kpt = gt_kpts[0]
                for k_idx, name in enumerate(kpt_names):
                    row[f'gt_{name}_X'] = g_kpt[k_idx, 0]
                    row[f'gt_{name}_Y'] = g_kpt[k_idx, 1]
                    row[f'gt_{name}_Z'] = g_kpt[k_idx, 2]
            if sensor_coords_str:
                row['selected_sensor_coords'] = sensor_coords_str
            
            predictions_for_csv.append(row)

def append_bbox_predictions(predictions_for_csv, res, target, device, sensor_coords_str):
    pred_boxes = res['boxes']
    pred_scores = res['scores']
    pred_labels = res['labels']

    image_path = target['image_path']
    true_label = target['labels'][0].item()

    true_box_cxcywhd_norm = target['boxes']
    true_box_xyzxyz_norm = box_cxcywhd_to_xyzxyz(true_box_cxcywhd_norm)
    
    h, w, d = target['orig_size']
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