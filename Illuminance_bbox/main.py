import argparse
import datetime
import json
import random
import time
from pathlib import Path
import csv
import sys
import pandas as pd
import functools
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000000, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--backbone', default='illuminance', help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_classes', default=3, type=int,help="Number of object classes (excluding the 'no object' class)")
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--center_loss_coef', default=3, type=float,
                        help="Weight for the center coordinate prediction loss in reduction mode")
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--gpu', default="0", type=str, help="GPU specific directory for output.")
    parser.add_argument('--dataset_file', default='illuminance')
    parser.add_argument('--num_sensors', type=int, default=36,
                        help="Number of illuminance sensors to use. -1 means all.")
    parser.add_argument('--sensor_dropout_rate', type=float, default=0.0,
                        help="Dropout rate for sensors during training (0.0 means no dropout).")
    parser.add_argument('--random_sensors', action='store_true',
                        help="If specified, train and val sets use different random sensor sets.")
    parser.add_argument('--data', default=None, nargs='+', type=str,
                        help="Name(s) of the dataset(s) to be split (e.g., 8:1:1). "
                             "If multiple names are provided, they will be split and concatenated.")
    parser.add_argument('--train', default=None, nargs='+', type=str,
                        help="Name(s) for the training dataset(s). If provided, --val and --test must also be provided (separate file mode).")
    parser.add_argument('--val', default=None, nargs='+', type=str,
                        help="Name(s) for the validation dataset(s). Used in separate file mode.")
    parser.add_argument('--test', default=None, nargs='+', type=str,
                        help="Name(s) for the test dataset(s). Used in separate file mode.")
    parser.add_argument('--exp_name', type=str, default=None,
                    help="Custom name for the output directory. If not provided, dataset names are joined.")
    parser.add_argument('--standardize_threshold', default=5.0, type=float,
                        help='Threshold for illuminance standardization. Values below this are ignored (treated as NaN).')
    
    parser.add_argument('--grid_size', default=6, type=int,
                        help="Size of the interpolation grid (grid_size x grid_size)")
    parser.add_argument('--window_size', type=int, default=41,
                        help="Number of illuminance frames to use as input window.")
    parser.add_argument('--sub_window_size', type=int, default=37,
                        help="Size of the sub-window for time-series slicing.")
    parser.add_argument('--stride', type=int, default=1,
                        help="Stride for time-series slicing.")
    parser.add_argument('--acc_weight', default=0.3, type=float,
                        help="Weight for accuracy in the best model metric (for validation)")
    parser.add_argument('--iou_weight', default=0.7, type=float,
                        help="Weight for IoU in the best model metric (for validation)")
    parser.add_argument('--early_stopping_patience', default=50, type=int,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument('--model_mode', type=str, default='time_sensor',
                        choices=['sensor', 'sensor_no_pe', 'sensor_interpolation', 'time', 'time_sensor',
                                 'time_sensor_no_time_pe', 'time_sensor_no_sensor_pe', 'time_sensor_no_pe',
                                 'time_sensor_wifi_like'],
                        help="Select the model architecture and experiment mode.")
    parser.add_argument('--environment', type=str, default='none',
                    choices=['none', 'PE', 'query', 'PE_query', 'sub', 'sequence'],
                    help="How to use the environment.csv data. 'PE': positional encoding. 'query': add to queries. 'sub': spectral subtraction.")
    parser.add_argument('--env_seq_len', type=int, default=333,
                    help="Fixed length that raw per-sensor environment.csv time series are resampled to for 'PE'/'query'/'sequence' modes (dataset environment.csv files have varying lengths).")
    parser.add_argument('--env_model_type', type=str, default='conv3d',
                    choices=['mlp', 'conv3d', 'spatial_gru'],
                    help="Feature extractor for the [env_seq_len, grid_size, grid_size] environment vector. "
                         "'mlp': flatten + 2-layer MLP. 'conv3d': joint space-time 3D convolution. "
                         "'spatial_gru': per-frame 2D conv followed by a GRU over time.")
    # parser.add_argument('--environment', type=str, default='none',
    #                     choices=['none', 'PE', 'query', 'sub', 'sequence'],
    #                     help="How to use the environment.csv data. 'PE': positional encoding. 'query': add to queries. 'sub': spectral subtraction.")
    parser.add_argument('--scale', type=str, default='full',
                        choices=['full', 'sparse', 'reductionCNNmse', 'reductionCNNconfidence', 'reductionVariance', 'reductionCNNgumbel', 'reductionAttention', 'reductionAttentiongumbel'],
                        help="Type of attention mechanism for 'time_sensor' mode.")
    parser.add_argument('--k_neighbors', type=int, default=36,
                        help="Number of nearest neighbors for sparse attention or for sensor selection in reduction mode.")
    parser.add_argument('--space_width', type=float, default=480.0,
                        help="The width of the physical space (e.g., in mm or pixels).")
    parser.add_argument('--space_height', type=float, default=480.0,
                        help="The height of the physical space (e.g., in mm or pixels).")
    parser.add_argument('--space_depth', type=float, default=480.0,
                        help="The depth (Z-axis) of the physical space.")
    parser.add_argument('--standardize', action='store_true',
                        help="If specified, standardize the illuminance sensor data based on the training set.")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--comment', type=str, default='',
                        help="A name or memo for this training run, to be saved in the log.")
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="Proportion of the dataset to use for training when splitting (default: 0.8).")
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help="Proportion of the dataset to use for validation when splitting (default: 0.1).")
    parser.add_argument('--n_runs', default=1, type=int,
                        help="Number of times to run the entire training/evaluation process for averaging.")
    return parser


def run_single_experiment(args):
    utils.init_distributed_mode(args)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    PRIMARY_NAS_PATH = "/nas/data_2/ueno1212/light"
    SECONDARY_NAS_PATH = "/home/hiroto.tsunoda/intern/light"
    
    if os.path.isdir(PRIMARY_NAS_PATH):
        nas_path = PRIMARY_NAS_PATH
        print(f"--- Using primary NAS path: {nas_path} ---")
    elif os.path.isdir(SECONDARY_NAS_PATH):
        nas_path = SECONDARY_NAS_PATH
        print(f"--- Primary NAS path not found. Using secondary NAS path: {nas_path} ---")
    else:
        nas_path = "."
        print(f"--- WARNING: Neither primary nor secondary NAS paths found. Using default path: {nas_path}. ---")

    if args.data and args.train:
        raise ValueError("Cannot specify both --data (for splitting) and --train (for separate files) simultaneously.")
    if (args.train or args.val or args.test) and not (args.train and args.val and args.test):
        raise ValueError("In separate file mode, --train, --val, and --test must ALL be specified.")
    if not args.data and not args.train:
        raise ValueError("You must specify either --data (for splitting) or --train, --val, --test (for separate files).")

    if args.train:
        print("--- Running in SEPARATE file mode (train/val/test) ---")
        args.do_split = False
        
        train_names = args.train
        val_names = args.val
        test_names = args.test
        
        args.train_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in train_names]
        args.train_bbox_path = [f'{nas_path}/data/{name}/bbox_label.csv' for name in train_names]
        args.train_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in train_names]
        
        args.val_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in val_names]
        args.val_bbox_path = [f'{nas_path}/data/{name}/bbox_label.csv' for name in val_names]
        args.val_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in val_names]

        args.test_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in test_names]
        args.test_bbox_path = [f'{nas_path}/data/{name}/bbox_label.csv' for name in test_names]
        args.test_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in test_names]
        
        args.standardization_train_light_paths = args.train_illuminance_path

        if args.exp_name:
            base_name = args.exp_name
        else:
            full_joined_name = "+".join(args.data)
            if len(full_joined_name) > 100:
                base_name = f"{args.data[0]}_and_{len(args.data)-1}_others"
            else:
                base_name = full_joined_name
        model_type_str = args.model_mode
        args.output_dir = f'{nas_path}/output/{base_name}/{args.scale}_{model_type_str}_3d/k_{args.k_neighbors}/{args.num_sensors}sensors'
        if args.random_sensors: args.output_dir += '_random'

    else:
        if args.train_ratio <= 0 or args.val_ratio <= 0:
            raise ValueError("train_ratio and val_ratio must be positive.")
        if args.train_ratio + args.val_ratio >= 1.0:
            raise ValueError(f"The sum of train_ratio ({args.train_ratio}) and val_ratio ({args.val_ratio}) must be less than 1.0.")
        
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        print(f"--- Running in SPLIT mode (using --data: {args.data}) ---")
        print(f"--- Splitting data with ratios: Train={args.train_ratio:.2f}, Val={args.val_ratio:.2f}, Test={test_ratio:.2f} ---")
        
        args.do_split = True

        args.data_illuminance_paths = [f'{nas_path}/data/{name}/light.csv' for name in args.data]
        args.data_bbox_paths = [f'{nas_path}/data/{name}/bbox_label.csv' for name in args.data]
        args.data_environment_paths = [f'{nas_path}/data/{name}/environment.csv' for name in args.data]
        args.train_illuminance_path = args.data_illuminance_paths
        args.train_bbox_path = args.data_bbox_paths
        args.val_illuminance_path = args.data_illuminance_paths
        args.val_bbox_path = args.data_bbox_paths
        args.test_illuminance_path = args.data_illuminance_paths
        args.test_bbox_path = args.data_bbox_paths
        args.train_environment_path = args.data_environment_paths
        args.val_environment_path = args.data_environment_paths
        args.test_environment_path = args.data_environment_paths
        args.standardization_train_light_paths = args.data_illuminance_paths

        if args.exp_name:
            base_name = args.exp_name
        else:
            full_joined_name = "+".join(args.data)
            if len(full_joined_name) > 100:
                base_name = f"{args.data[0]}_and_{len(args.data)-1}_others"
            else:
                base_name = full_joined_name
        model_type_str = args.model_mode
        args.output_dir = f'{nas_path}/output/{base_name}/{args.scale}_{model_type_str}_3d/k_{args.k_neighbors}/{args.num_sensors}sensors'
        if args.random_sensors: args.output_dir += '_random'

    base_output_dir = Path(args.output_dir)
    gpu_output_dir = base_output_dir / f'GPU{args.gpu}'
    gpu_output_dir.mkdir(parents=True, exist_ok=True)
    args.gpu_output_dir = gpu_output_dir
    
    if args.standardize:
        print("--- Standardization enabled. Calculating stats from training data... ---")
        try:
            train_paths = args.standardization_train_light_paths
            df_list = []
            for p in train_paths:
                df = pd.read_csv(p)
                
                if args.do_split:
                    num_train = int(len(df) * args.train_ratio)
                    original_len = len(df)
                    df = df.iloc[:num_train]
                    print(f"   -> [Standardization] Using first {num_train}/{original_len} rows from {os.path.basename(p)}")
                
                df_list.append(df)
            
            full_train_df = pd.concat(df_list, ignore_index=True)
            
            sensor_columns = [col for col in full_train_df.columns if col != 'timestamp']
            sensor_df = full_train_df[sensor_columns]
            
            if args.standardize_threshold > 0:
                print(f"   -> Applying standardization threshold: values < {args.standardize_threshold} are ignored.")
                sensor_df = sensor_df[sensor_df >= args.standardize_threshold]
            
            mean_vals = sensor_df.mean()
            std_vals = sensor_df.std()
            
            args.mean = torch.tensor(mean_vals.values, dtype=torch.float32)
            args.std = torch.tensor(std_vals.values, dtype=torch.float32)
            print("   -> Stats calculation complete.")
            
        except FileNotFoundError as e:
            print(f"ERROR: Could not read training data for standardization: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during standardization setup: {e}")
            sys.exit(1)
    else:
        args.mean = None
        args.std = None

    if args.do_split:
        header_ref_path = args.train_illuminance_path[0]
    else:
        header_ref_path = args.train_illuminance_path[0]
        
    try:
        all_sensor_columns = pd.read_csv(header_ref_path, nrows=0).columns.tolist()[1:]
    except FileNotFoundError:
        print(f"ERROR: Could not read header from reference file: {header_ref_path}")
        sys.exit(1)
    
    if 0 < args.num_sensors < len(all_sensor_columns):
        args.actual_num_sensors = args.num_sensors
    else:
        args.actual_num_sensors = len(all_sensor_columns)

    if not args.random_sensors:
        print("--- Running in SHARED sensor mode. ---")
        if args.actual_num_sensors < len(all_sensor_columns):
            args.selected_columns = random.sample(all_sensor_columns, args.actual_num_sensors)
            print(args.selected_columns)
        else:
            args.selected_columns = all_sensor_columns
    else:
        print("--- Running in RANDOM sensor mode for train/val. ---")

    print(f"--- Number of sensors to use: {args.actual_num_sensors} / {len(all_sensor_columns)} ---")

    print(args)

    device = torch.device(args.device)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    
    reduction_cnn_params = [p for n, p in model_without_ddp.named_parameters() 
                            if "reduction_cnn" in n and p.requires_grad]
    
    confidence_cnn_params = [p for n, p in model_without_ddp.named_parameters() 
                             if "confidence_cnn" in n and p.requires_grad]

    env_query_cnn_params = [p for n, p in model_without_ddp.named_parameters()
                                if "env_query_cnn" in n and p.requires_grad]

    backbone_params = [p for n, p in model_without_ddp.named_parameters() 
                       if "backbone" in n and "reduction_cnn" not in n and "confidence_cnn" not in n and p.requires_grad]
    
    base_params = [p for n, p in model_without_ddp.named_parameters() 
                   if "backbone" not in n and "env_query_cnn" not in n and p.requires_grad]

    cnn_lr = args.lr * 10

    param_dicts = [
        {"params": base_params},
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": reduction_cnn_params, "lr": cnn_lr},
        {"params": confidence_cnn_params, "lr": cnn_lr},
        {"params": env_query_cnn_params, "lr": cnn_lr},
    ]

    print(f"Optimizer setup: Base LR={args.lr}, Backbone LR={args.lr_backbone}, CNNs LR={cnn_lr}")

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    collate_train = functools.partial(utils.collate_fn, args=args, split='train')
    collate_val   = functools.partial(utils.collate_fn, args=args, split='val')
    collate_test  = functools.partial(utils.collate_fn, args=args, split='test')

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_val, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=collate_test, num_workers=args.num_workers)

    if args.dataset_file == "illuminance":
        from datasets import illuminance as illuminance_dataset
        base_ds = None

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            print(f"--- Loaded pretrained model weights from {args.resume}. Starting finetuning from epoch 0. ---")

    if args.eval:
        test_stats, _, _, _ = evaluate(model, criterion, postprocessors,
                                                     data_loader_val, base_ds, device, gpu_output_dir)
        
        return

    best_metric_score = -1.0
    patience_counter = 0
    ACC_WEIGHT = args.acc_weight
    IOU_WEIGHT = args.iou_weight
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    best_checkpoint_filename = f'best_{run_timestamp}.pth'
    best_checkpoint_path = gpu_output_dir / best_checkpoint_filename
    last_checkpoint_filename = f'last_{run_timestamp}.pth'
    last_checkpoint_path = gpu_output_dir / last_checkpoint_filename

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        val_stats, _, val_preds_df, val_coord_df = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, gpu_output_dir,
            header_prefix='Val'
        )
        if 'accuracy' in val_stats and 'iou' in val_stats and 'f1_score' in val_stats:
            current_acc = val_stats.get('accuracy', 0.0)
            current_iou = val_stats.get('iou', 0.0)
            current_f1 = val_stats.get('f1_score', 0.0)
            
            current_metric = (current_acc + current_iou + current_f1) / 3.0
            
            print(f"Epoch {epoch}: Current average metric (Acc+IoU+F1)/3: {current_metric:.4f} (Best: {best_metric_score:.4f})")
            print(f"  (Details: Acc: {current_acc:.4f}, IoU: {current_iou:.4f}, F1: {current_f1:.4f})")

            if current_metric > best_metric_score:
                print(f"  New best model found! Saving checkpoint and validation predictions.\n")
                best_metric_score = current_metric
                patience_counter = 0

                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, best_checkpoint_path)

            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}\n")
            
            if not val_coord_df.empty:
                val_coord_df.to_csv(gpu_output_dir / 'val_coord_predictions_latest.csv', index=False)
            val_preds_df.to_csv(gpu_output_dir / 'val_prediction_latest.csv', index=False)
        print(f"Now Train: {gpu_output_dir}")
        print(f"Comment: {args.comment}\n")
        

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            log_file_path = gpu_output_dir / "log.csv"
            log_txt_path = gpu_output_dir / "log.txt"
            
            fieldnames = [
                'epoch', 
                'train_iou', 'train_f1_score', 'train_accuracy',
                'train_CNN_error', 
                'val_iou', 'val_f1_score', 'val_accuracy', 
                'val_CNN_error'
            ]
            
            remaining_keys = sorted([k for k in log_stats.keys() if k not in fieldnames])
            fieldnames.extend(remaining_keys)

            with open(log_file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if epoch == args.start_epoch:
                    writer.writeheader()
                writer.writerow(log_stats)

            with open(log_txt_path, "a") as f:
                if epoch == args.start_epoch:
                    f.write("="*80 + "\n")
                    f.write(f"START TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("--- ARGS ---\n")
                    for k, v in vars(args).items():
                        f.write(f"{k}: {v}\n")
                    f.write("\n")
                    f.write("--- DATASETS ---\n")
                    if args.do_split:
                        f.write(f"Mode: Splitting single dataset (8:1:1)\n")
                        f.write(f"Source Dataset: {args.train}\n")
                    else:
                        f.write(f"Mode: Using separate files\n")
                        f.write(f"  Training:   {args.train}\n")
                        f.write(f"  Validation: {args.val}\n")
                        f.write(f"  Test:       {args.test}\n")
                    f.write("\n")
                    f.write("--- SENSOR CONFIG (TRAIN) ---\n")
                    train_sensors = dataset_train.selected_columns
                    f.write(f"Using {len(train_sensors)} sensors:\n")
                    for i in range(0, len(train_sensors), 3):
                        f.write("  " + "  ".join(f"{col:<20}" for col in train_sensors[i:i+3]) + "\n")
                    f.write("\n")
                    f.write("--- SENSOR CONFIG (VAL) ---\n")
                    val_sensors = dataset_val.selected_columns
                    f.write(f"Using {len(val_sensors)} sensors:\n")
                    for i in range(0, len(val_sensors), 3):
                        f.write("  " + "  ".join(f"{col:<20}" for col in val_sensors[i:i+3]) + "\n")
                    f.write("\n")
                    f.write("="*80 + "\n\n")
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
            print(f"\n--- Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement. ---\n")
            break

    print(f"Saving last model (epoch {epoch}) to {last_checkpoint_path}...")
    utils.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }, last_checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print("\n--- Starting final testing phase with the best model ---")
    
    final_log_stats = None
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Best model weights loaded.")

        test_stats, _, _, test_coord_df = evaluate(
            model, criterion, postprocessors, data_loader_test, base_ds, device, gpu_output_dir,
            prediction_filename=f'test_prediction_{run_timestamp}.csv'
        )
        
        final_log_stats = {f'test_{k}': v for k, v in test_stats.items()}
        desired_key_order = [
            'ap', 'ap50', 'ap75', 
            'iou', 'f1_score', 'accuracy', 
            'loss_ce', 'loss_bbox', 'loss_giou', 
            'class_error', 'CNN_error', 'loss_center'
        ]

        print("Test Stats:")
        ordered_stats_str_list = []
        
        for key in desired_key_order:
            test_key = f'test_{key}'
            if test_key in final_log_stats:
                value = final_log_stats[test_key]
                if isinstance(value, float):
                    ordered_stats_str_list.append(f"'{test_key}': {value:.4f}")
                else:
                    ordered_stats_str_list.append(f"'{test_key}': {value}")
        
        for key, value in final_log_stats.items():
            if key not in [f'test_{k}' for k in desired_key_order]:
                if isinstance(value, float):
                    ordered_stats_str_list.append(f"'{key}': {value:.4f}")
                else:
                    ordered_stats_str_list.append(f"'{key}': {value}")

        final_stats_str = "{ " + ", ".join(ordered_stats_str_list) + " }"
        print(final_stats_str)

        if args.output_dir and utils.is_main_process():
            with open(gpu_output_dir / "log.txt", "a") as f:
                f.write("\n--- EXPERIMENTS CONDITION ---\n")
                f.write(f"{args.comment}: {i + 1} / {args.n_runs}\n")
                f.write("\n--- FINAL TEST RESULTS ---\n")
                f.write(final_stats_str + "\n")
                f.write("prediction_timestamp:" + str(run_timestamp) + "\n")
                f.write("output_path:" + str(gpu_output_dir) + "\n")
                f.write("\n--- SENSOR CONFIG (TEST) ---\n")
                test_sensors = dataset_test.selected_columns
                f.write(f"Using {len(test_sensors)} sensors:\n")
                for i in range(0, len(test_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in test_sensors[i:i+3]) + "\n")
                f.write("\n")
    else:
        print(f"Best model checkpoint not found at {best_checkpoint_path}. Skipping final test.")

    print("\n--- Starting final testing phase with the LAST model ---")
    
    if last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Last model weights loaded.")

        test_stats_last, _, _, test_coord_df_last = evaluate(
            model, criterion, postprocessors, data_loader_test, base_ds, device, gpu_output_dir,
            prediction_filename=f'test_prediction_last_{run_timestamp}.csv'
        )
        
        final_log_stats.update({f'last_test_{k}': v for k, v in test_stats_last.items()})

        print("Last Model Test Stats:")
        ordered_stats_str_list = []
        keys_to_show = [
            'ap', 'ap50', 'ap75', 
            'iou', 'f1_score', 'accuracy', 
            'loss_ce', 'loss_bbox', 'loss_giou', 
            'class_error', 'CNN_error', 'loss_center'
        ]
        for k in keys_to_show:
            if k in test_stats_last:
                val = test_stats_last[k]
                if isinstance(val, float):
                    ordered_stats_str_list.append(f"'{k}': {val:.4f}")
                else:
                    ordered_stats_str_list.append(f"'{k}': {val}")
        
        last_stats_str = "{ " + ", ".join(ordered_stats_str_list) + " }"
        print(last_stats_str)

        if args.output_dir and utils.is_main_process():
            with open(gpu_output_dir / "log.txt", "a") as f:
                f.write("\n--- LAST MODEL TEST RESULTS ---\n")
                f.write(last_stats_str + "\n")
                f.write(f"checkpoint: {last_checkpoint_filename}\n")
    else:
        print(f"Last model checkpoint not found at {last_checkpoint_path}.")

    return final_log_stats

def log_average_results(all_final_stats: list, base_output_dir: Path, args):
    """Calculates and logs the average of final test results over all runs."""
    
    if not all_final_stats:
        print("No valid stats collected. Skipping average logging.")
        return

    n_runs = len(all_final_stats)
    print("\n" + "="*80)
    print(f"--- AVERAGING RESULTS ACROSS {n_runs} RUNS ---")
    print("="*80 + "\n")

    df = pd.DataFrame(all_final_stats)
    
    avg_stats = df.mean()
    std_stats = df.std()
    
    header_str = f"--- Average Test Results ({n_runs} runs) ---"
    
    avg_stats_lines = []
    std_stats_lines = []
    
    desired_key_order = [
        'test_accuracy', 'test_iou', 'test_f1_score', 
        'test_ap', 'test_ap50', 'test_ap75', 
        'last_test_accuracy', 'last_test_iou', 'last_test_f1_score', 
        'last_test_ap', 'last_test_ap50', 'last_test_ap75', 
    ]
    for c in range(args.num_classes):
            desired_key_order.extend([
                f'AP_class_{c}', 
                f'AP50_class_{c}', 
                f'AP75_class_{c}'
            ])
    desired_key_order.extend([
        'test_loss_ce', 'test_loss_bbox', 'test_loss_giou', 
        'test_class_error', 'test_CNN_error', 'test_loss_center'
    ])
    
    sorted_keys = [key for key in desired_key_order if key in avg_stats]
    remaining_keys = [key for key in avg_stats.keys() if key not in sorted_keys]
    sorted_keys.extend(remaining_keys)

    for key in sorted_keys:
        avg_val = avg_stats[key]
        std_val = std_stats[key]
        avg_stats_lines.append(f"  '{key}': {avg_val:.4f},")
        std_stats_lines.append(f"  '{key}': {std_val:.4f},")

    avg_stats_str = "{ 'Average Stats': {\n" + "\n".join(avg_stats_lines) + "\n} }"
    std_stats_str = "{ 'Std Deviation': {\n" + "\n".join(std_stats_lines) + "\n} }"
    
    full_log_message = f"{header_str}\n{avg_stats_str}\n{std_stats_str}\n"

    print(full_log_message)

    log_txt_path = base_output_dir / "log.txt"
    if log_txt_path.exists():
        with open(log_txt_path, "a") as f:
            f.write("\n\n" + "="*80 + "\n")
            f.write(full_log_message)
            f.write("="*80 + "\n")
    
    log_csv_path = base_output_dir / "log.csv"
    if log_csv_path.exists():
        try:
            avg_stats_dict = avg_stats.to_dict()
            avg_stats_dict['epoch'] = f'AVG_{n_runs}_RUNS'
            
            std_stats_dict = std_stats.to_dict()
            std_stats_dict = {f"{k}_std": v for k, v in std_stats_dict.items()}
            
            csv_row_data = {**avg_stats_dict, **std_stats_dict}

            with open(log_csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                fieldnames = next(reader)
            
            new_fieldnames = list(fieldnames)
            for key in csv_row_data.keys():
                if key not in new_fieldnames:
                    new_fieldnames.append(key)
                    
            with open(log_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
                
                if set(fieldnames) != set(new_fieldnames):
                    pass
                
                writer.writerow(csv_row_data)
                
        except Exception as e:
            print(f"Warning: Could not append average to log.csv. Error: {e}")

def main(args):
    """
    --n_runs で指定された回数だけ実験を実行し、最後に平均をログに出力する
    """
    all_final_stats = []
    base_output_dir = None
    original_seed = args.seed

    for i in range(args.n_runs):
        print("\n" + "="*80)
        print(f"--- STARTING RUN {i + 1} / {args.n_runs} ---")
        print("="*80 + "\n")
        
        args.seed = original_seed + i 
        print(f"--- Setting seed for this run to: {args.seed} ---")
        
        single_run_stats = run_single_experiment(args)
        
        if single_run_stats:
            all_final_stats.append(single_run_stats)
        
        if i == 0 and hasattr(args, 'gpu_output_dir'):
             base_output_dir = args.gpu_output_dir

        print("\n" + "="*80)
        print(f"--- COMPLETED RUN {i + 1} / {args.n_runs} ---")
        print("="*80 + "\n")

    if all_final_stats and base_output_dir:
        log_average_results(all_final_stats, base_output_dir, args)
    elif args.n_runs > 1:
        print(f"Could not calculate averages. Found {len(all_final_stats)} valid run(s).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)