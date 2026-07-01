import argparse
import torch
import pandas as pd
from mmcv.cnn import get_model_complexity_info
import functools

from main import get_args_parser
from models import build_model
from datasets import build_dataset
import util.misc as utils

class ModelWrapper(torch.nn.Module):
    """
    mmcvがキーワード引数でモデルを呼び出すのを吸収するためのラッパークラス。
    forward(**kwargs) で受け取ったキーワード引数を一つの辞書にまとめて、
    元のモデルの forward(samples) に渡す。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(kwargs)

def analyze_model_complexity(args):
    """
    指定された引数でモデルを構築し、mmcvのflops_counterで解析します。
    """
    model, _, _ = build_model(args)
    model.eval()
    
    wrapped_model = ModelWrapper(model)

    try:
        dataset_val = build_dataset(image_set='val', args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        collate_val = functools.partial(utils.collate_fn, args=args, split='val') 
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, args.batch_size, sampler=sampler_val,
            collate_fn=collate_val, num_workers=args.num_workers
        )
        samples, _ = next(iter(data_loader_val))
    except Exception as e:
        print(f"\nError: データセットの構築またはサンプル取得に失敗しました。データセットのパスを確認してください。")
        print(f"詳細: {e}")
        return None, None
    input_constructor = lambda _: samples
    flops_str, params_str = get_model_complexity_info(
        wrapped_model,
        input_shape=(1,),
        input_constructor=input_constructor,
        print_per_layer_stat=False,
        as_strings=True
    )
    return flops_str, params_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MMCV FLOPs Counter Script for DETR-Light', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if not args.train:
        print("Error: データセット名が必要です。--train引数を指定してください。")
        exit()
    nas_path = "/nas/data_2/tsunoda/light"
    train_names = args.train if isinstance(args.train, list) else [args.train]
    val_names = args.val if args.val else train_names
    
    args.train_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in train_names]
    args.train_bbox_path = [f'{nas_path}/data/{name}/bbox_label.csv' for name in train_names]
    args.train_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in train_names]
    
    args.val_illuminance_path = [f'{nas_path}/data/{name}/light.csv' for name in val_names]
    args.val_bbox_path = [f'{nas_path}/data/{name}/bbox_label.csv' for name in val_names]
    args.val_environment_path = [f'{nas_path}/data/{name}/environment.csv' for name in val_names]
    
    args.do_split = False
    try:
        all_sensor_columns = pd.read_csv(args.train_illuminance_path[0], nrows=0).columns.tolist()[1:]
        if 0 < args.num_sensors < len(all_sensor_columns):
            args.actual_num_sensors = args.num_sensors
        else:
            args.actual_num_sensors = len(all_sensor_columns)
    except FileNotFoundError:
        print(f"Error: 照度CSVファイルが見つかりません: {args.train_illuminance_path}")
        exit()

    args.mean = None
    args.std = None
    print("\n" + "="*60)
    print(f"Analyzing configuration:")
    print(f"  --scale: {args.scale}")
    print(f"  --num_sensors: {args.num_sensors}")
    if 'reduction' in args.scale:
        print(f"  --k_neighbors: {args.k_neighbors}")
    print("="*60)
    
    flops, params = analyze_model_complexity(args)

    if flops and params:
        print("\n" + "="*60)
        print("           Complexity Analysis Result")
        print("="*60)
        print(f"  - Parameters: {params}")
        print(f"  - FLOPs:      {flops}")
        print("="*60)