import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import random
import sys

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

def box_xyzxyz_to_cxcywhd(x):
    """ (x1,y1,z1,x2,y2,z2) -> (cx,cy,cz,w,h,d) """
    x1, y1, z1, x2, y2, z2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
         (x2 - x1), (y2 - y1), (z2 - z1)]
    return torch.stack(b, dim=-1)

class IlluminanceDetectionDataset(Dataset):
    def __init__(self, illuminance_csv_path, bbox_csv_path, environment_csv_path, args, split='train', time_format='%Y-%m-%d %H:%M:%S %f', do_split=False):
        assert split in ['train', 'val', 'test']
        self.train_print_count = 0

        is_list_input = isinstance(illuminance_csv_path, list)
        
        merged_df_split = None
        ill_dfs_all_concat = []
        self.environment_vectors_raw = []
        self.all_env_sensor_names = []
        
        current_ill_index_offset = 0

        if do_split:
            print(f"--- [Dataset: {split}] Loading, merging, and splitting dataset(s) individually... ---")
            
            if not is_list_input:
                illuminance_csv_path = [illuminance_csv_path]
                bbox_csv_path = [bbox_csv_path]
                environment_csv_path = [environment_csv_path]

            merged_splits_to_concat = []

            for ill_path, bbox_path, env_path in zip(illuminance_csv_path, bbox_csv_path, environment_csv_path):
                try:
                    ill_df_single = pd.read_csv(ill_path)
                    bbox_df_single = pd.read_csv(bbox_path)
                except FileNotFoundError as e:
                    print(f"ERROR: Could not find file {e.filename}. Skipping this file.")
                    continue

                ill_df_single.dropna(inplace=True)
                bbox_df_single.dropna(inplace=True)
                if len(ill_df_single) == 0 or len(bbox_df_single) == 0:
                    print(f"Warning: Skipping {ill_path} (or bbox) due to empty data after NaN drop.")
                    continue

                if args.environment != 'none':
                    try:
                        env_df_single = pd.read_csv(env_path)
                        env_df_single.dropna(inplace=True)
                        all_env_sensors = env_df_single.columns.tolist()[1:]
                        if not all_env_sensors:
                             raise ValueError(f"No sensor columns found in {env_path}")
                        
                        current_env_data = env_df_single[all_env_sensors].values.astype(np.float32).T
                        self.environment_vectors_raw.append(torch.from_numpy(current_env_data))
                        self.all_env_sensor_names.append(all_env_sensors)
                        
                    except FileNotFoundError:
                        print(f"ERROR: Environment file {env_path} not found. Mode '{args.environment}' requires it.")
                    except Exception as e:
                        print(f"ERROR: Failed to process environment file {env_path}: {e}")
                        sys.exit(1)
                
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
                
                n_samples_merged = len(merged_df_single)
                num_train = int(n_samples_merged * args.train_ratio)
                num_val = int(n_samples_merged * args.val_ratio)

                if split == 'train':
                    split_df = merged_df_single.iloc[:num_train]
                elif split == 'val':
                    split_df = merged_df_single.iloc[num_train : num_train + num_val]
                else:
                    split_df = merged_df_single.iloc[num_train + num_val:]
                
                merged_splits_to_concat.append(split_df)
                
                ill_dfs_all_concat.append(ill_df_single)
                current_ill_index_offset += len(ill_df_single)

            if not merged_splits_to_concat:
                 raise ValueError(f"No valid data loaded for split '{split}'. Check file paths and NaN values.")
            
            merged_df_split = pd.concat(merged_splits_to_concat, ignore_index=True)
            print(f"   -> [Dataset: {split}] Using {len(merged_df_split)} rows (combined).")

        else:
            print(f"--- [Dataset: {split}] Loading, merging, and concatenating dataset(s)... ---")
            
            if not is_list_input:
                illuminance_csv_path = [illuminance_csv_path]
                bbox_csv_path = [bbox_csv_path]
                environment_csv_path = [environment_csv_path]

            merged_files_to_concat = []

            for ill_path, bbox_path, env_path in zip(illuminance_csv_path, bbox_csv_path, environment_csv_path):
                try:
                    ill_df_single = pd.read_csv(ill_path)
                    bbox_df_single = pd.read_csv(bbox_path)
                except FileNotFoundError as e:
                    print(f"ERROR: Could not find file {e.filename}. Skipping this file.")
                    continue

                ill_df_single.dropna(inplace=True)
                bbox_df_single.dropna(inplace=True)
                if len(ill_df_single) == 0 or len(bbox_df_single) == 0:
                    print(f"Warning: Skipping {ill_path} (or bbox) due to empty data after NaN drop.")
                    continue

                if args.environment != 'none':
                    try:
                        env_df_single = pd.read_csv(env_path)
                        env_df_single.dropna(inplace=True)
                        all_env_sensors = env_df_single.columns.tolist()[1:]
                        if not all_env_sensors:
                             raise ValueError(f"No sensor columns found in {env_path}")
                        
                        current_env_data = env_df_single[all_env_sensors].values.astype(np.float32).T
                        self.environment_vectors_raw.append(torch.from_numpy(current_env_data))
                        self.all_env_sensor_names.append(all_env_sensors)
                        
                    except FileNotFoundError:
                        print(f"ERROR: Environment file {env_path} not found. Mode '{args.environment}' requires it.")
                        sys.exit(1)
                    except Exception as e:
                        print(f"ERROR: Failed to process environment file {env_path}: {e}")
                        sys.exit(1)
                
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
                
                merged_files_to_concat.append(merged_df_single)
                
                ill_dfs_all_concat.append(ill_df_single)
                current_ill_index_offset += len(ill_df_single)

            if not merged_files_to_concat:
                 raise ValueError(f"No valid data loaded for split '{split}'. Check file paths and NaN values.")
            
            merged_df_split = pd.concat(merged_files_to_concat, ignore_index=True)
            print(f"   -> [Dataset: {split}] Using {len(merged_df_split)} rows (combined).")

        if not ill_dfs_all_concat:
             raise ValueError("No illuminance data loaded.")
        self.illuminance_df = pd.concat(ill_dfs_all_concat, ignore_index=True)

        initial_light_rows = len(self.illuminance_df)
        self.illuminance_df.dropna(inplace=True)
        dropped_light_rows = initial_light_rows - len(self.illuminance_df)
        if dropped_light_rows > 0:
            print(f"--- Dropped {dropped_light_rows} rows from illuminance data due to NaN.")
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
            print(f"   -> Filtering standardization stats for {len(self.selected_columns)} selected sensors.")
            
            all_columns_map = {col: i for i, col in enumerate(all_sensor_columns)}
            selected_indices = [all_columns_map[col] for col in self.selected_columns]
            
            self.mean = self.mean[selected_indices]
            self.std = self.std[selected_indices]

        self.std_eps = self.std + 1e-6 if self.std is not None else None
        if args.environment != 'none':
            print(f"--- Aligning environment vectors to {len(self.selected_columns)} selected sensors... (Mode: {args.environment}) ---")
            self.environment_vectors = []
            
            all_selected_columns_map = {col: i for i, col in enumerate(self.selected_columns)}
            
            if args.environment == 'sub':
                print(f"    -> Calculating average noise spectrum (W={args.window_size})...")
                W = args.window_size
                FreqBins = W // 2 + 1
                
                for i, (raw_vec_tensor, raw_names) in enumerate(zip(self.environment_vectors_raw, self.all_env_sensor_names)):
                    raw_map = {name: k for k, name in enumerate(raw_names)}
                    N_selected = len(self.selected_columns)
                    
                    aligned_spectrum = torch.zeros(N_selected, FreqBins, dtype=torch.float32)
                    
                    missing_cols_count = 0
                    for col_name, selected_idx in all_selected_columns_map.items():
                        if col_name in raw_map:
                            raw_idx = raw_map[col_name]
                            
                            sensor_time_series = raw_vec_tensor[raw_idx, :]
                            
                            num_windows = sensor_time_series.shape[0] - W + 1
                            if num_windows <= 0:
                                print(f"Warning: Env file {i}, sensor {col_name} has insufficient data ({sensor_time_series.shape[0]} < {W}). Skipping sensor.")
                                continue
                                
                            sensor_windows = sensor_time_series.unfold(0, W, 1)
                            
                            avg_sensor_spectrum = torch.fft.rfft(sensor_windows, dim=1).abs().mean(dim=0)
                            
                            aligned_spectrum[selected_idx] = avg_sensor_spectrum
                        else:
                            missing_cols_count += 1
                            
                    if missing_cols_count > 0:
                        print(f"   -> Warning: Dataset ID {i}: {missing_cols_count} selected sensors were not found in its environment.csv for 'sub' mode.")
                    self.environment_vectors.append(aligned_spectrum)
            
            else:
                env_seq_len = args.env_seq_len
                grid_size = args.grid_size
                print(f"    -> Using raw environment vectors as a {grid_size}x{grid_size} spatial grid, resampled to fixed length {env_seq_len}...")

                for i, (raw_vec_tensor, raw_names) in enumerate(zip(self.environment_vectors_raw, self.all_env_sensor_names)):
                    raw_map = {name: k for k, name in enumerate(raw_names)}

                    # environment.csv files have different lengths across datasets, so
                    # resample every raw vector to a shared fixed length for stacking/model input.
                    raw_vec_resampled = F.interpolate(
                        raw_vec_tensor.unsqueeze(0), size=env_seq_len, mode='linear', align_corners=False
                    ).squeeze(0)

                    # [T, H, W]: sensors are placed at their (x, y) grid position so the
                    # 3D CNN can read space and time jointly.
                    aligned_vector = torch.zeros(env_seq_len, grid_size, grid_size, dtype=torch.float32)

                    missing_cols_count = 0
                    for col_name in all_selected_columns_map:
                        if col_name in raw_map:
                            raw_idx = raw_map[col_name]
                            gx, gy = (float(v) for v in col_name.split(','))
                            row = round(gx * (grid_size - 1))
                            col = round(gy * (grid_size - 1))
                            aligned_vector[:, row, col] = raw_vec_resampled[raw_idx]
                        else:
                            missing_cols_count += 1

                    if missing_cols_count > 0:
                        print(f"   -> Warning: Dataset ID {i}: {missing_cols_count} selected sensors were not found in its environment.csv for '{args.environment}' mode.")
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
            raise ValueError(
                "Sensor coordinates in the CSV header are not normalized to the [0, 1] range. "
                "Please ensure all coordinate values are pre-normalized."
            )
        print("\n" + "="*50)
        print(f"Dataset initialized. Using the following {len(self.selected_columns)} sensor columns:")
        for i in range(0, len(self.selected_columns), 3):
             print("  ".join(f"{col:<20}" for col in self.selected_columns[i:i+3]))
        print("="*50 + "\n")

        self.samples = []
        for (ill_center_idx, source_id), group in merged_df_split.groupby(['global_ill_index', 'source_dataset_id']):
            start_idx = ill_center_idx - (args.window_size - 1)
            end_idx = ill_center_idx + 1

            if start_idx < 0:
                continue

            illuminance_window = self.illuminance_df.iloc[start_idx:end_idx][self.selected_columns].values

            if illuminance_window.shape[0] != args.window_size:
                continue

            boxes_for_frame = []
            labels_for_frame = []
            centers_for_frame = []
            has_center_coords = 'center_x' in group.columns and 'center_y' in group.columns
            space_w, space_h, space_d = args.space_width, args.space_height, args.space_depth

            for _, row in group.iterrows():
                box = torch.tensor([
                    row['x1'], row['y1'], row['z1'],
                    row['x2'], row['y2'], row['z2']
                ])

                box[[0, 3]] /= space_w
                box[[1, 4]] /= space_h
                box[[2, 5]] /= space_d
                
                boxes_for_frame.append(box)
                labels_for_frame.append(int(row['label']))
                if has_center_coords:
                    centers_for_frame.append(torch.tensor([row['center_x'], row['center_y']], dtype=torch.float32))

            boxes_xyzxyz = torch.stack(boxes_for_frame)
            boxes_cxcywhd = box_xyzxyz_to_cxcywhd(boxes_xyzxyz)
            
            labels = torch.tensor(labels_for_frame, dtype=torch.long)

            first_row = group.iloc[0]
            target = {
                'boxes': boxes_cxcywhd,
                'labels': labels,
                'image_id': torch.tensor([ill_center_idx]),
                'orig_size': torch.tensor([space_h, space_w, space_d]),
                'image_path': first_row['image_path']
            }
            if has_center_coords and centers_for_frame:
                target['center_coords'] = centers_for_frame[0]
            if args.environment != 'none':
                target['env_vector'] = self.environment_vectors[source_id]

            if split == 'train' and self.train_print_count < 1:
                print(f"  [Debug] Adding train sample {self.train_print_count + 1}: "
                      f"BBox Timestamp: {first_row['timestamp']} | "
                      f"Mapped Illuminance Idx: {ill_center_idx}")
            self.train_print_count += 1

            self.samples.append((illuminance_window.astype(np.float32), target))
        print(f"  [Debug] Adding train sample {self.train_print_count + 1}: "
                f"BBox Timestamp: {first_row['timestamp']} | "
                f"Mapped Illuminance Idx: {ill_center_idx}")
        print(self.train_print_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        illuminance_data, target = self.samples[idx]

        illuminance_tensor = torch.from_numpy(illuminance_data).T 

        if self.mean is not None and self.std_eps is not None:
            illuminance_tensor = (illuminance_tensor - self.mean.unsqueeze(1)) / self.std_eps.unsqueeze(1)

        return illuminance_tensor, self.sensor_coords, target

def build(image_set, args):
    if image_set == 'train':
        illuminance_path = args.train_illuminance_path
        bbox_path = args.train_bbox_path
        environment_path = args.train_environment_path
    elif image_set == 'val':
        illuminance_path = args.val_illuminance_path
        bbox_path = args.val_bbox_path
        environment_path = args.val_environment_path
    elif image_set == 'test':
        illuminance_path = args.test_illuminance_path
        bbox_path = args.test_bbox_path
        environment_path = args.test_environment_path
    else:
        raise ValueError(f"Unknown image_set: {image_set}")
    
    dataset = IlluminanceDetectionDataset(
        illuminance_csv_path=illuminance_path, 
        bbox_csv_path=bbox_path,
        environment_csv_path=environment_path,
        args=args,
        do_split=args.do_split,
        split=image_set
    )
    return dataset