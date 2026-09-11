import argparse
import datetime
import random
from pathlib import Path
import csv
import fnmatch
import os
import functools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from util.logger import log_results
from datasets.dataset import build as build_dataset
from engine import evaluate, train
from models.detr import build as build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector and pose estimator', add_help=False)
    
    parser.add_argument('--task', default='bbox', type=str, choices=['bbox', 'pose'])

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=10, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.3, type=float)
    
    # for bbox
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--remove_difficult', action='store_true')
    
    # for pose
    parser.add_argument('--keypoint_loss_coef', default=10, type=float)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_runs', default=1, type=int)
    parser.add_argument('--early_stopping_patience', default=50, type=int)

    parser.add_argument('--num_sensors', type=int, default=36)
    parser.add_argument('--sensor_dropout_rate', type=float, default=0.0)
    parser.add_argument('--random_sensors', action='store_true')
    
    parser.add_argument('--window_size', type=int, default=41)
    parser.add_argument('--sub_window_size', type=int, default=37)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--k_neighbors', type=int, default=36)

    parser.add_argument('--data', default=None, nargs='+', type=str)
    parser.add_argument('--train', default=None, nargs='+', type=str)
    parser.add_argument('--val', default=None, nargs='+', type=str)
    parser.add_argument('--test', default=None, nargs='+', type=str)
    parser.add_argument('--exp_name', type=str, default=None)
    
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    parser.add_argument('--standardize', action='store_false')
    parser.add_argument('--standardize_threshold', default=5.0, type=float)
    parser.add_argument('--space_width', type=float, default=480.0)
    parser.add_argument('--space_height', type=float, default=480.0)
    parser.add_argument('--space_depth', type=float, default=480.0)
    parser.add_argument('--real_space_size', nargs=3, type=float, default=[2500.0, 2500.0, 2500.0])

    parser.add_argument('--model_mode', type=str, default='time_sensor', 
                        choices=['time', 'time_sensor', 'time_sensor_no_time_pe', 
                                 'time_sensor_no_sensor_pe', 'time_sensor_no_pe', 'time_sensor_wifi_like'])
    parser.add_argument('--scale', type=str, default='full', 
                        choices=['full', 'reductionCNNconfidence', 'reductionVariance', 'reductionAttention'])
    parser.add_argument('--environment', type=str, default='none', choices=['none', 'EE', 'EQ', 'PE_query'])
    return parser

def run(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    PRIMARY_NAS_PATH = "/nas/data_2/ueno1212/light"
    SECONDARY_NAS_PATH = ""
    
    if os.path.isdir(PRIMARY_NAS_PATH):
        nas_path = PRIMARY_NAS_PATH
    elif os.path.isdir(SECONDARY_NAS_PATH):
        nas_path = SECONDARY_NAS_PATH
    else:
        nas_path = "."
        print(f"WARNING: Neither primary nor secondary NAS paths found. Using default path: {nas_path}.")

    task = args.task
    label_csv_name = "pose.csv" if task == "pose" else "bbox_label.csv"
    out_dir_prefix = "output_pose" if task == "pose" else "output_bbox"

    if args.train:
        print(f"train: {args.train}\nval  : {args.val}\ntest : {args.test}")
        args.do_split = False
        args.train_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in args.train]
        args.train_bbox_path = [f'{nas_path}/data/{name}/{label_csv_name}' for name in args.train]
        args.train_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in args.train]
        
        args.val_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in args.val]
        args.val_bbox_path = [f'{nas_path}/data/{name}/{label_csv_name}' for name in args.val]
        args.val_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in args.val]

        args.test_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in args.test]
        args.test_bbox_path = [f'{nas_path}/data/{name}/{label_csv_name}' for name in args.test]
        args.test_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in args.test]
        
    else:
        print(f"train/val/test: {args.data}")
        args.do_split = True
        data_ill_paths = [f'{nas_path}/data/{name}/light.csv' for name in args.data]
        data_bbox_paths = [f'{nas_path}/data/{name}/{label_csv_name}' for name in args.data]
        data_env_paths = [f'{nas_path}/data/{name}/environment.csv' for name in args.data]
        
        args.train_illuminance_path = args.val_illuminance_path = args.test_illuminance_path = data_ill_paths
        args.train_bbox_path = args.val_bbox_path = args.test_bbox_path = data_bbox_paths
        args.train_environment_path = args.val_environment_path = args.test_environment_path = data_env_paths

    output_dir = f'{nas_path}/{out_dir_prefix}/{args.exp_name}/{args.scale}_{args.model_mode}/k_{getattr(args, "k_neighbors", "none")}/{args.num_sensors}sensors'
    if args.random_sensors: output_dir += '_random'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.standardize:
        train_paths = args.train_illuminance_path
        df_list = []
        for p in train_paths:
            df = pd.read_csv(p)
            
            if args.do_split:
                num_train = int(len(df) * args.train_ratio)
                df = df.iloc[:num_train]
            
            df_list.append(df)
        
        full_train_df = pd.concat(df_list, ignore_index=True)
        sensor_columns = [col for col in full_train_df.columns if col != 'timestamp']
        sensor_df = full_train_df[sensor_columns]
        
        if args.standardize_threshold > 0:
            sensor_df = sensor_df[sensor_df >= args.standardize_threshold]
        
        mean_vals = sensor_df.mean()
        std_vals = sensor_df.std()
        
        args.mean = torch.tensor(mean_vals.values, dtype=torch.float32)
        args.std = torch.tensor(std_vals.values, dtype=torch.float32)

    else:
        args.mean = None
        args.std = None

    if args.standardize and args.environment != 'none':
        env_train_paths = args.train_environment_path
        env_df_list = []
        for p in env_train_paths:
            df = pd.read_csv(p)

            if args.do_split:
                num_train = int(len(df) * args.train_ratio)
                df = df.iloc[:num_train]

            env_df_list.append(df)

        full_train_env_df = pd.concat(env_df_list, ignore_index=True)
        env_columns = [col for col in full_train_env_df.columns if col != 'timestamp']
        env_df = full_train_env_df[env_columns]

        args.env_column_names = env_columns
        args.env_mean = torch.tensor(env_df.mean().values, dtype=torch.float32)
        args.env_std = torch.tensor(env_df.std().values, dtype=torch.float32)

    else:
        args.env_column_names = None
        args.env_mean = None
        args.env_std = None

    all_sensor_columns = pd.read_csv(args.train_illuminance_path[0], nrows=0).columns.tolist()[1:]
    
    if 0 < args.num_sensors < len(all_sensor_columns):
        args.actual_num_sensors = args.num_sensors
    else:
        args.actual_num_sensors = len(all_sensor_columns)

    if not args.random_sensors:
        print(f"Running in Identical Sensor Mode: {args.actual_num_sensors} / {len(all_sensor_columns)}")
        if args.actual_num_sensors < len(all_sensor_columns):
            args.selected_columns = random.sample(all_sensor_columns, args.actual_num_sensors)
        else:
            args.selected_columns = all_sensor_columns
    else:
        print(f"Running in Layout-Agnostic Sensor Mode: {args.actual_num_sensors} / {len(all_sensor_columns)}")

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    confidence_cnn_params = [p for n, p in model_without_ddp.named_parameters() 
                             if "confidence_cnn" in n and p.requires_grad]
    backbone_params = [p for n, p in model_without_ddp.named_parameters() 
                       if "backbone" in n and "reduction_cnn" not in n and "confidence_cnn" not in n and p.requires_grad]
    base_params = [p for n, p in model_without_ddp.named_parameters()
                   if "backbone" not in n and "env_query_cnn" not in n and p.requires_grad]
    eq_cnn_params = [p for n, p in model_without_ddp.named_parameters()
                                if "env_query_cnn" in n and p.requires_grad]

    cnn_lr = args.lr * 10
    param_dicts = [
        {"params": base_params},
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": confidence_cnn_params, "lr": cnn_lr},
        {"params": eq_cnn_params, "lr": cnn_lr},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    
    collate_train = functools.partial(utils.collate_fn, args=args, split='train')
    collate_val   = functools.partial(utils.collate_fn, args=args, split='val')
    collate_test  = functools.partial(utils.collate_fn, args=args, split='test')

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_train)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=collate_val)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False, collate_fn=collate_test)

    if args.eval:
        evaluate(model, criterion, postprocessors, data_loader_val, device, output_dir, args=args)
        return

    best_metric_score = -1000.0
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    best_checkpoint_path = f"{output_dir}/best_{run_timestamp}.pth"

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args=args)
        lr_scheduler.step()

        val_stats = evaluate(model, criterion, postprocessors, data_loader_val, device, output_dir, prediction_filename=f'val_prediction_latest.csv', header_prefix="Val", args=args)
        
        if 'accuracy' in val_stats and 'f1_score' in val_stats:
            current_acc = val_stats.get('accuracy', 0.0)
            current_f1 = val_stats.get('f1_score', 0.0)
            
            if task == 'pose':
                current_mpjpe = val_stats.get('mpjpe', 0.0)
                current_score = -current_mpjpe/100.0 + current_acc + current_f1
                print(f"Epoch {epoch}: Score (-MPJPE/100+Acc+F1): {current_score:.4f} (Best: {best_metric_score:.4f})")
            else:
                current_iou = val_stats.get('iou', 0.0)
                current_score = (current_acc + current_iou + current_f1) / 3.0
                print(f"Epoch {epoch}: Score (Acc+IoU+F1)/3: {current_score:.4f} (Best: {best_metric_score:.4f})")

            if current_score > best_metric_score:
                print("  New best model found! Saving checkpoint.")
                best_metric_score = current_score
                patience_counter = 0
                torch.save({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'args': args}, best_checkpoint_path)
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        print(f"Now Train: {output_dir}\n\n"+"-"*30+"\n")
        
        log_stats = {**{f'tr_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': v for k, v in val_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        base_metrics = ['mpjpe', 'mpjdle_x', 'mpjdle_y', 'mpjdle_z', 'f1_score', 'accuracy'] if task == 'pose' else ['iou', 'f1_score', 'accuracy']
        fieldnames = ['epoch'] + [f'tr_{m}' for m in base_metrics] + [f'val_{m}' for m in base_metrics]
        remaining_keys = sorted([k for k in log_stats.keys() if k not in fieldnames])
        fieldnames.extend(remaining_keys)

        with open(f"{output_dir}/log.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if epoch == args.start_epoch: writer.writeheader()
            writer.writerow(log_stats)

        EXCLUDE_ARGS = {"mean", "std", "actual_num_sensors", "*illuminance_path*", "*bbox_path*", "t_runs"}
        with open(f"{output_dir}/log.txt", "a") as f:
            if epoch == args.start_epoch:
                f.write("\n" + "="*80 + "\n")
                f.write(f"START TIME: {run_timestamp}\n")
                f.write(f"RUN       : {args.t_runs} / {args.n_runs}\n")
                f.write("--- ARGS ---\n")
                filtered_args = [
                    (k, v)
                    for k, v in vars(args).items()
                    if not any(fnmatch.fnmatch(k, pattern) for pattern in EXCLUDE_ARGS)
                ]
                max_len = max([len(k) for k, _ in filtered_args], default=0)
                for k, v in filtered_args:
                    f.write(f"  {k:<{max_len}} : {v}\n")
                f.write("\n")
                f.write("--- DATASETS ---\n")
                if args.do_split:
                    f.write(f" Train:Val:Test = {args.train_ratio:.2f}:{args.val_ratio:.2f}:{1 - args.train_ratio - args.val_ratio:.2f}\n")
                    f.write(f" Source Dataset: {args.data}\n")
                else:
                    f.write(f" Train: {args.train}\n")
                    f.write(f" Val  : {args.val}\n")
                    f.write(f" Test : {args.test}\n")
                f.write("\n")
                f.write("--- SENSOR CONFIG ---\n")
                train_sensors = dataset_train.selected_columns
                f.write(f"[train] Using {len(train_sensors)} sensors:\n")
                for i in range(0, len(train_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in train_sensors[i:i+3]) + "\n")
                f.write("\n")
                val_sensors = dataset_val.selected_columns
                f.write(f"[val] Using {len(val_sensors)} sensors:\n")
                for i in range(0, len(val_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in val_sensors[i:i+3]) + "\n")
                f.write("\n")
                test_sensors = dataset_test.selected_columns
                f.write(f"[test] Using {len(test_sensors)} sensors:\n")
                for i in range(0, len(test_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in test_sensors[i:i+3]) + "\n")
                f.write("\n")
                header = "".join([f"{key:<18}" for key in fieldnames])
                f.write(header + "\n")

            row_str = ""
            for key in fieldnames:
                value = log_stats.get(key)
                if isinstance(value, float):
                    row_str += f"{value:<18.5f}"
                else:
                    row_str += f"{str(value):<18}"
            f.write(row_str + "\n")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.\n")
            break

    final_log_stats = None
    if Path(best_checkpoint_path).exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        test_stats = evaluate(model, criterion, postprocessors, data_loader_test, device, output_dir, prediction_filename=f'test_prediction_{run_timestamp}.csv', header_prefix="Test", args=args)
        final_log_stats = {f'{k}': v for k, v in test_stats.items()}
        
        desired_key_order = ['mpjpe', 'mpjdle_x', 'f1_score', 'accuracy', 'loss_ce', 'loss_keypoints'] if task == 'pose' else ['ap', 'ap50', 'ap75', 'iou', 'f1_score', 'accuracy', 'loss_ce', 'loss_bbox']
        
        ordered_stats = [f"'{test_key}': {final_log_stats.get(test_key, 0):.4f}" for k in desired_key_order if (test_key := f'{k}') in final_log_stats]
        final_stats_str = ", ".join(ordered_stats)

        with open(f"{output_dir}/log.txt", "a") as f:
            f.write("\n--- TEST RESULTS ---\n")
            f.write(final_stats_str + "\n")
            f.write("run_timestamp:" + str(run_timestamp) + "\n")
            f.write("output_path  :" + str(output_dir) + "\n")
            f.write("\n")
        
    return final_log_stats, output_dir

def main(args):
    all_final_stats = []
    original_seed = args.seed

    for i in range(args.n_runs):
        print(f"{'='*10} Run {i+1}/{args.n_runs} {'='*10}")
        args.seed = original_seed + i
        args.t_runs = i+1

        single_run_stats, output_dir = run(args)
        
        if single_run_stats:
            all_final_stats.append(single_run_stats)
        
        if i == 0: base_output_dir = output_dir

    if all_final_stats and base_output_dir:
        log_results(all_final_stats, base_output_dir, args)
    elif args.n_runs > 1:
        print(f"Could not calculate averages. Found {len(all_final_stats)} valid run(s).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)