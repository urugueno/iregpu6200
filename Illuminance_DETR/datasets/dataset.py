import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import random

from util.calculate import box_xyzxyz_to_cxcywhd

class IlluminanceDetectionDataset(Dataset):
    KEYPOINT_NAMES = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    OUTPUT_KEYPOINT_NAMES = KEYPOINT_NAMES + ["neck"]

    def __init__(self, illuminance_csv_path, bbox_csv_path, environment_csv_path, args, split='train', time_format='%Y-%m-%d %H:%M:%S %f', do_split=False):
        assert split in ['train', 'val', 'test']
        
        self.task = args.task

        is_list_input = isinstance(illuminance_csv_path, list)
        merged_df_split = None
        ill_dfs_all_concat = []
        self.environment_vectors_raw = []
        self.all_env_sensor_names = []
        current_ill_index_offset = 0

        if self.task == 'pose':
            required_tgt_cols = ['timestamp']
            for name in self.KEYPOINT_NAMES:
                required_tgt_cols.extend([f'{name}_X', f'{name}_Y', f'{name}_Z'])
        else:
            required_tgt_cols = ['timestamp', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'label']

        if not is_list_input:
            illuminance_csv_path = [illuminance_csv_path]
            bbox_csv_path = [bbox_csv_path]
            environment_csv_path = [environment_csv_path]

        merged_files_to_concat = []

        for source_dataset_id, (ill_path, bbox_path, env_path) in enumerate(zip(illuminance_csv_path, bbox_csv_path, environment_csv_path)):
            try:
                ill_df_single = pd.read_csv(ill_path)
                bbox_df_single = pd.read_csv(bbox_path)
            except FileNotFoundError as e:
                print(f"ERROR: Could not d file {e.filename}. Skipping this file.")
                continue

            ill_df_single.dropna(inplace=True)
            
            check_subset = [c for c in required_tgt_cols if c in bbox_df_single.columns]
            bbox_df_single.dropna(subset=check_subset, inplace=True)
            
            if len(ill_df_single) == 0 or len(bbox_df_single) == 0:
                print(f"Warning: Skipping {ill_path} (or bbox) due to empty data after NaN drop.")
                continue

            if args.environment != 'none':
                env_df_single = pd.read_csv(env_path)
                env_df_single.dropna(inplace=True)
                all_env_sensors = env_df_single.columns.tolist()[1:]
                if not all_env_sensors:
                        raise ValueError(f"No sensor columns found in {env_path}")
                
                current_env_data = env_df_single[all_env_sensors].values.astype(np.float32).T
                self.environment_vectors_raw.append(torch.from_numpy(current_env_data))
                self.all_env_sensor_names.append(all_env_sensors)
                    
            source_dataset_id = len(self.environment_vectors_raw) - 1 if args.environment != 'none' else 0

            ill_df_single['timestamp_dt'] = pd.to_datetime(ill_df_single['timestamp'], format=time_format)
            bbox_df_single['timestamp_dt'] = pd.to_datetime(bbox_df_single['timestamp'], format=time_format)
            
            ill_df_single = ill_df_single.sort_values('timestamp_dt').reset_index(drop=True)
            ill_df_single['global_ill_index'] = ill_df_single.index + current_ill_index_offset

            merged_df_single = pd.merge_asof(
                bbox_df_single.sort_values('timestamp_dt'),
                ill_df_single[['timestamp_dt', 'global_ill_index']],
                on='timestamp_dt',
                direction='nearest'
            )
            merged_df_single['source_dataset_id'] = source_dataset_id
            
            if do_split:
                n_samples_merged = len(merged_df_single)
                num_train = int(n_samples_merged * args.train_ratio)
                num_val = int(n_samples_merged * args.val_ratio)

                if split == 'train':
                    split_df = merged_df_single.iloc[:num_train]
                elif split == 'val':
                    split_df = merged_df_single.iloc[num_train : num_train + num_val]
                else:
                    split_df = merged_df_single.iloc[num_train + num_val:]
                merged_files_to_concat.append(split_df)
            else:
                merged_files_to_concat.append(merged_df_single)
            
            ill_dfs_all_concat.append(ill_df_single)
            current_ill_index_offset += len(ill_df_single)

        if not merged_files_to_concat:
             raise ValueError(f"No valid data loaded for split '{split}'. Check file paths and NaN values.")
        
        merged_df_split = pd.concat(merged_files_to_concat, ignore_index=True)

        if not ill_dfs_all_concat:
             raise ValueError("No illuminance data loaded.")
        self.illuminance_df = pd.concat(ill_dfs_all_concat, ignore_index=True)

        initial_light_rows = len(self.illuminance_df)
        self.illuminance_df.dropna(inplace=True)
        if initial_light_rows - len(self.illuminance_df) > 0:
            print(f"Dropped {initial_light_rows - len(self.illuminance_df)} rows from illuminance data due to NaN.")
            
        exclude_cols = ['timestamp', 'timestamp_dt', 'global_ill_index']
        all_sensor_columns = [col for col in self.illuminance_df.columns if col not in exclude_cols]
        
        if hasattr(args, 'selected_columns'):
            self.selected_columns = args.selected_columns
        else:
            if 0 < args.actual_num_sensors < len(all_sensor_columns):
                self.selected_columns = random.sample(all_sensor_columns, args.actual_num_sensors)
            else:
                self.selected_columns = all_sensor_columns
        
        self.mean = args.mean
        self.std = args.std

        if self.mean is not None and self.std is not None and len(all_sensor_columns) != len(self.selected_columns):
            all_columns_map = {col: i for i, col in enumerate(all_sensor_columns)}
            selected_indices = [all_columns_map[col] for col in self.selected_columns]
            self.mean = self.mean[selected_indices]
            self.std = self.std[selected_indices]

        self.std_eps = self.std + 1e-6 if self.std is not None else None

        if args.environment != 'none':
            self.environment_vectors = []
            all_selected_columns_map = {col: i for i, col in enumerate(self.selected_columns)}
            N_selected = len(self.selected_columns)

            for i, (raw_vec_tensor, raw_names) in enumerate(zip(self.environment_vectors_raw, self.all_env_sensor_names)):
                raw_vec_mean = raw_vec_tensor.mean(dim=1)

                raw_map = {name: k for k, name in enumerate(raw_names)}

                aligned_vector = torch.zeros(N_selected, dtype=torch.float32)

                for col_name, selected_idx in all_selected_columns_map.items():
                    if col_name in raw_map:
                        raw_idx = raw_map[col_name]
                        aligned_vector[selected_idx] = raw_vec_mean[raw_idx]

                self.environment_vectors.append(aligned_vector)

        header_ref_path = illuminance_csv_path[0] if is_list_input else illuminance_csv_path
        try:
            header = pd.read_csv(header_ref_path, nrows=0).columns.tolist()[1:]
        except FileNotFoundError:
             raise ValueError(f"Could not read header from reference file: {header_ref_path}")
        coords = []
        all_coords_map = {name: list(map(float, name.split(','))) for name in header}
        for col_name in self.selected_columns:
            coords.append(all_coords_map[col_name])
        self.sensor_coords = torch.tensor(coords, dtype=torch.float32)
        
        if torch.any(self.sensor_coords < 0) or torch.any(self.sensor_coords > 1):
            raise ValueError("Sensor coordinates are not normalized to [0, 1].")

        print(f"\n[{split}] Using the following {len(self.selected_columns)} sensor columns:")
        for i in range(0, len(self.selected_columns), 4):
             print("  ".join(f"{col:<20}" for col in self.selected_columns[i:i+4]))
        print("")

        self.samples = []
        space_w, space_h, space_d = args.space_width, args.space_height, args.space_depth

        for (ill_center_idx, source_id), group in merged_df_split.groupby(['global_ill_index', 'source_dataset_id']):
            start_idx = ill_center_idx - (args.window_size - 1)
            end_idx = ill_center_idx + 1

            if start_idx < 0:
                continue

            illuminance_window = self.illuminance_df.iloc[start_idx:end_idx][self.selected_columns].values
            if illuminance_window.shape[0] != args.window_size:
                continue

            target = {}
            labels_for_frame = []
            
            if self.task == 'pose':
                keypoints_for_frame = []
                for _, row in group.iterrows():
                    kpts = [[row[f'{name}_X'], row[f'{name}_Y'], row[f'{name}_Z']] for name in self.KEYPOINT_NAMES]
                    kpts_tensor = torch.tensor(kpts, dtype=torch.float32)
                    
                    neck = (kpts_tensor[1] + kpts_tensor[2]) / 2.0
                    kpts_tensor = torch.cat([kpts_tensor, neck.unsqueeze(0)], dim=0)

                    kpts_tensor[:, 0] /= space_w
                    kpts_tensor[:, 1] /= space_h
                    kpts_tensor[:, 2] /= space_d
                    
                    keypoints_for_frame.append(kpts_tensor)
                    labels_for_frame.append(0)

                target['keypoints'] = torch.stack(keypoints_for_frame) if keypoints_for_frame else torch.zeros((0, 14, 3), dtype=torch.float32)
                target['labels'] = torch.tensor(labels_for_frame, dtype=torch.long)
                
            else:
                boxes_for_frame = []
                for _, row in group.iterrows():
                    box = torch.tensor([row['x1'], row['y1'], row['z1'], row['x2'], row['y2'], row['z2']])
                    box[[0, 3]] /= space_w
                    box[[1, 4]] /= space_h
                    box[[2, 5]] /= space_d
                    boxes_for_frame.append(box)
                    labels_for_frame.append(int(row['label']))

                boxes_xyzxyz = torch.stack(boxes_for_frame) if boxes_for_frame else torch.zeros((0, 6), dtype=torch.float32)
                target['boxes'] = box_xyzxyz_to_cxcywhd(boxes_xyzxyz) if len(boxes_xyzxyz) > 0 else boxes_xyzxyz
                target['labels'] = torch.tensor(labels_for_frame, dtype=torch.long)

            first_row = group.iloc[0]
            target['image_id'] = torch.tensor([ill_center_idx])
            target['orig_size'] = torch.tensor([space_h, space_w, space_d])
            target['image_path'] = first_row.get('image_path', first_row.get('color_filepath', ''))
            if args.environment != 'none':
                target['env_vector'] = self.environment_vectors[source_id]

            self.samples.append((illuminance_window.astype(np.float32), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        illuminance_data, target = self.samples[idx]
        illuminance_tensor = torch.from_numpy(illuminance_data).T 

        if self.mean is not None and self.std_eps is not None:
            illuminance_tensor = (illuminance_tensor - self.mean.unsqueeze(1)) / self.std_eps.unsqueeze(1)

        return illuminance_tensor, self.sensor_coords, target

def build(image_set, args):
    illuminance_path = getattr(args, f"{image_set}_illuminance_path")
    bbox_path = getattr(args, f"{image_set}_bbox_path")
    environment_path = getattr(args, f"{image_set}_environment_path")
    
    return IlluminanceDetectionDataset(
        illuminance_csv_path=illuminance_path, 
        bbox_csv_path=bbox_path,
        environment_csv_path=environment_path,
        args=args,
        do_split=args.do_split,
        split=image_set
    )