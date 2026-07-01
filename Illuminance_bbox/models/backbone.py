"""
Backbone modules.
"""
from collections import OrderedDict
import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from scipy.spatial.distance import cdist

from util.misc import NestedTensor, is_main_process

from .position_encoding import PositionEmbeddingFromCoords, PositionEmbeddingTime1D
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        nested_tensor = self[0](tensor_list)
        
        pos = self[1](nested_tensor)
        
        return [nested_tensor], [pos]
    


class SensorCNN(nn.Module):
    """
    全センサーの時系列データから中心座標 (x, y) を予測するCNN。
    入力形状: (batch, sequence_length, num_features)
    """
    def __init__(self, num_features, sequence_length):
        super(SensorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        flattened_size = 128 * (sequence_length // 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class ConfidenceCNN(nn.Module):
    """
    全センサーの時系列データから、各センサーの確信度スコアを予測するCNN。
    出力: (batch, num_features)
    """
    def __init__(self, num_features, sequence_length):
        super(ConfidenceCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * sequence_length, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.0):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

class GumbelSelectorCNN(nn.Module):
    """
    全センサーの時系列データから、各センサーの選択ロジットを予測するCNN。
    ConfidenceCNNとほぼ同じだが、最後のSigmoidがない（Raw Logitsを出力する）。
    出力: (batch, num_features)
    """
    def __init__(self, num_features, sequence_length):
        super(GumbelSelectorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * sequence_length, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class IlluminanceBackbone(nn.Module):
    def __init__(self, window_size, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(window_size, hidden_dim)
        self.position_embedding = PositionEmbeddingFromCoords(hidden_dim // 2)
        self.num_channels = hidden_dim

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        coords = samples_dict['coords']

        features = self.projection(illuminance)
        pos = self.position_embedding(coords)

        features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)
        pos_reshaped = pos.permute(0, 2, 1).unsqueeze(-1)

        mask = torch.zeros((features.shape[0], features.shape[1]), dtype=torch.bool, device=features.device)

        feature_nested_tensor = NestedTensor(features_reshaped, mask)

        return [feature_nested_tensor], [pos_reshaped]

class DirectSensorBackbone(nn.Module):
    def __init__(self, window_size, hidden_dim, args, model_mode='sensor'):
        super().__init__()
        self.args = args
        self.model_mode = model_mode
        self.projection = nn.Linear(window_size, hidden_dim)
        self.num_channels = hidden_dim

        if self.model_mode == 'sensor':
            self.position_embedding = PositionEmbeddingFromCoords(hidden_dim // 2)

        self.env_projection = None
        if self.args.environment == 'PE':
            self.env_projection = nn.Linear(self.args.actual_num_sensors, hidden_dim)

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        B, N, W = illuminance.shape
        
        features = self.projection(illuminance)

        if self.model_mode == 'sensor':
            coords = samples_dict['coords']
            pos = self.position_embedding(coords)
        elif self.model_mode == 'sensor_no_pe':
            pos = torch.zeros_like(features)

        if self.args.environment == 'PE':
            env_vector = samples_dict['env_vector']
            env_pe = self.env_projection(env_vector)
            
            pos = pos + env_pe.unsqueeze(1)
        
        features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)
        pos_reshaped = pos.permute(0, 2, 1).unsqueeze(-1)
        mask = samples_dict.get('sensor_padding_mask', 
                                torch.zeros((B, N), dtype=torch.bool, device=features.device))
        feature_nested_tensor = NestedTensor(features_reshaped, mask)
        
        return [feature_nested_tensor], [pos_reshaped]

class InterpolationBackbone(nn.Module):
    """
    センサーデータと座標を元に、補間によって固定サイズのグリッド特徴マップを生成するバックボーン
    """
    def __init__(self, window_size, hidden_dim, grid_size=16, idw_power=2):
        super().__init__()
        self.projection = nn.Linear(window_size, hidden_dim)
        self.grid_size = grid_size
        self.idw_power = idw_power
        self.num_channels = hidden_dim

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, grid_size),
            torch.linspace(0, 1, grid_size),
            indexing='xy'
        )
        self.grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        coords = samples_dict['coords']

        B, N, _ = illuminance.shape
        C = self.num_channels

        sensor_features = self.projection(illuminance)

        grid = self.grid_points.to(illuminance.device)
        G = grid.shape[0]

        dist_sq = torch.sum((coords.unsqueeze(1) - grid.unsqueeze(0).unsqueeze(2))**2, dim=-1)
        
        epsilon = 1e-6
        
        weights = 1.0 / (torch.pow(dist_sq, self.idw_power / 2) + epsilon)
        
        weights = F.normalize(weights, p=1, dim=2)

        interpolated_features = torch.bmm(weights, sensor_features)

        tolerance_sq = 1e-10 
        
        is_occupied_mask = dist_sq < tolerance_sq
        if is_occupied_mask.any():
            batch_indices, grid_indices, sensor_indices = torch.nonzero(is_occupied_mask, as_tuple=True)
            source_features = sensor_features[batch_indices, sensor_indices]
            interpolated_features[batch_indices, grid_indices] = source_features
        feature_map = interpolated_features.view(B, self.grid_size, self.grid_size, C).permute(0, 3, 1, 2)
        mask = torch.zeros((B, self.grid_size, self.grid_size), dtype=torch.bool, device=feature_map.device)
        return NestedTensor(feature_map, mask)

class TimeSequenceBackbone(nn.Module):
    def __init__(self, window_size, sub_window_size, stride, num_sensors, hidden_dim, args):
        super().__init__()
        self.args = args
        self.sub_window_size = sub_window_size
        self.stride = stride
        input_dim = sub_window_size * num_sensors
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        self.position_embedding = PositionEmbeddingTime1D(num_pos_feats=hidden_dim)
        self.num_channels = hidden_dim
        self.seq_len = math.floor((window_size - sub_window_size) / stride) + 1

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        B, N, W = illuminance.shape
        sliced = illuminance.unfold(2, self.sub_window_size, self.stride)
        sliced = sliced.permute(0, 2, 1, 3).flatten(2)
        features = self.projection(sliced)
        pos = self.position_embedding(features)
        features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)
        pos_reshaped = pos.permute(0, 2, 1).unsqueeze(-1)
        SeqLen = features.shape[1]
        mask = torch.zeros((B, SeqLen, 1), dtype=torch.bool, device=features.device)
        feature_nested_tensor = NestedTensor(features_reshaped, mask)
        
        return [feature_nested_tensor], [pos_reshaped]
    
class FeatureBasedSelector(nn.Module):
    """
    各センサの時系列特徴と、センサ間の関係性(Attention)に基づいて
    重要なセンサを選択するための軽量モジュール
    """
    def __init__(self, input_dim, hidden_dim, num_sensors, return_logits=False):
        super().__init__()
        self.sensor_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, hidden_dim)
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.return_logits = return_logits

    def forward(self, x):
        B, N, T = x.shape
        x_reshaped = x.view(B * N, 1, T)
        feats = self.sensor_encoder(x_reshaped)
        feats = feats.view(B, N, -1)
        scores = self.score_head(feats).squeeze(-1)
        
        if self.return_logits:
            return scores
        else:
            return torch.sigmoid(scores)
    
class TimeSensorBackbone(nn.Module):
    def __init__(self, sub_window_size, num_sensors, hidden_dim, args, scale='full', k=8, model_mode='time_sensor'):
        super().__init__()

        self.args = args
        self.projection = nn.Linear(sub_window_size, hidden_dim)
        self.temporal_pe_generator = PositionEmbeddingTime1D(num_pos_feats=hidden_dim)
        
        self.model_mode = model_mode
        if self.model_mode == 'time_sensor_wifi_like':
            self.spatial_pe_generator = PositionEmbeddingTime1D(num_pos_feats=hidden_dim)
        else:
            self.spatial_pe_generator = PositionEmbeddingFromCoords(num_pos_feats=hidden_dim // 2)
        
        self.num_channels = hidden_dim
        self.scale = scale
        self.k = k
        self.num_sensors = num_sensors
        self.register_buffer('cached_attn_mask', None, persistent=False)

        self.model_mode = model_mode

        self.env_projection = None
        if self.args.environment == 'PE':
            self.env_projection = nn.Linear(num_sensors, hidden_dim)

        self.reduction_cnn = None
        if self.scale == 'reductionCNNmse':
            print(f"--- Reduction CNN (MSE) enabled. Predicting coordinates and selecting k={self.k} sensors. ---")
            self.reduction_cnn = SensorCNN(
                num_features=self.num_sensors,
                sequence_length=sub_window_size
            )
        self.feature_selector = None
        if self.scale == 'reductionAttention':
            print(f"--- Reduction (Attention-based) enabled. Selecting top-k={self.k} sensors using Feature Context. ---")
            self.feature_selector = FeatureBasedSelector(
                input_dim=sub_window_size,
                hidden_dim=32,
                num_sensors=num_sensors,
                return_logits=False
            )
        if self.scale == 'reductionAttentiongumbel':
            print(f"--- Reduction (Attention + Gumbel) enabled. Selecting top-k={self.k} sensors with differentiable sampling. ---")
            self.feature_selector = FeatureBasedSelector(
                input_dim=sub_window_size,
                hidden_dim=32,
                num_sensors=num_sensors,
                return_logits=True
            )
        self.confidence_cnn = None
        if self.scale == 'reductionCNNconfidence':
            print(f"--- Reduction CNN (Confidence) enabled. Predicting scores and selecting top-k={self.k} sensors. ---")
            self.confidence_cnn = ConfidenceCNN(
                num_features=self.num_sensors,
                sequence_length=sub_window_size
            )
        self.gumbel_selector_cnn = None
        if self.scale == 'reductionCNNgumbel':
            print(f"--- Reduction CNN (Gumbel-Softmax) enabled. Selecting top-k={self.k} sensors with differentiable sampling. ---")
            self.gumbel_selector_cnn = GumbelSelectorCNN(
                num_features=self.num_sensors,
                sequence_length=sub_window_size
            )
        elif self.scale == 'reductionVariance':
            print(f"--- Reduction (Variance) enabled. Selecting top-k={self.k} sensors based on variance. ---")

        if self.scale == 'sparse':
            print(f"--- Sparse attention enabled with k={self.k} nearest neighbors. ---")
            if self.k >= self.num_sensors:
                print(f"Warning: k_neighbors ({self.k}) is >= num_sensors ({self.num_sensors}). Falling back to full attention.")
                self.scale = 'full'

    def _create_sparse_attn_mask(self, coords: torch.Tensor, seq_len: int, device: torch.device):
        """kNNに基づいてセンサー間のAttentionを制限するマスクを生成する"""
        if self.cached_attn_mask is not None:
            return self.cached_attn_mask

        print("--- Creating and caching sparse attention mask... ---")
        sensor_coords = coords[0].cpu().numpy()
        knn = NearestNeighbors(n_neighbors=self.k, algorithm='auto')
        knn.fit(sensor_coords)
        adj_matrix = knn.kneighbors_graph(sensor_coords).toarray()
        
        adj_matrix = adj_matrix + np.eye(self.num_sensors)
        adj_matrix = (adj_matrix > 0).astype(float)
        spatial_mask = np.kron(np.ones((seq_len, seq_len)), adj_matrix)
        temporal_mask = np.kron(np.eye(seq_len), np.ones((self.num_sensors, self.num_sensors)))
        final_mask_logic = (spatial_mask > 0) | (temporal_mask > 0)
        attn_mask = torch.from_numpy(~final_mask_logic).to(device)
        
        self.cached_attn_mask = attn_mask
        return attn_mask

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        coords = samples_dict['coords']
        B, N, W = illuminance.shape
        C = self.num_channels
        env_pe = None
        if self.args.environment == 'PE':
            env_vector = samples_dict['env_vector']
            env_pe = self.env_projection(env_vector)
        predicted_center_coords = None
        selected_sensor_indices = None

        original_sensor_padding_mask = samples_dict.get('sensor_padding_mask', None)
        sensor_padding_mask = original_sensor_padding_mask

        if self.scale == 'reductionCNNmse':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            cnn_input = cnn_input_window.permute(0, 2, 1)
            
            predicted_center_coords = self.reduction_cnn(cnn_input)
            all_sensor_coords = coords[0]
            dist = torch.cdist(predicted_center_coords.unsqueeze(1), all_sensor_coords.unsqueeze(0)).squeeze(1)
            _, top_k_indices = torch.topk(dist, self.k, dim=1, largest=False)
            selected_sensor_indices = top_k_indices
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance = illuminance[batch_indices, top_k_indices, :]
            coords = coords[batch_indices, top_k_indices, :]
            N = self.k
            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

        elif self.scale == 'reductionCNNgumbel':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            cnn_input = cnn_input_window.permute(0, 2, 1)
            
            logits = self.gumbel_selector_cnn(cnn_input)
            
            if self.training:
                tau = 1.0 
                scores = gumbel_softmax_sample(logits, temperature=tau)
            else:
                scores = F.softmax(logits, dim=-1)
            top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance_selected = illuminance[batch_indices, top_k_indices, :]
            coords_selected = coords[batch_indices, top_k_indices, :]
            illuminance = illuminance_selected * top_k_scores.unsqueeze(-1)
            coords = coords_selected
            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k

        elif self.scale == 'reductionCNNconfidence':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            cnn_input = cnn_input_window.permute(0, 2, 1)
            
            confidence_scores = self.confidence_cnn(cnn_input)
            _, top_k_indices = torch.topk(confidence_scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance = illuminance[batch_indices, top_k_indices, :]
            coords = coords[batch_indices, top_k_indices, :]
            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k

        elif self.scale == 'reductionAttentiongumbel':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            logits = self.feature_selector(cnn_input_window)
            if self.training:
                tau = 1.0
                scores = gumbel_softmax_sample(logits, temperature=tau)
            else:
                scores = F.softmax(logits, dim=-1)
            top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance_selected = illuminance[batch_indices, top_k_indices, :]
            illuminance = illuminance_selected * top_k_scores.unsqueeze(-1)
            
            coords = coords[batch_indices, top_k_indices, :]

            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k

        elif self.scale == 'reductionAttention':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            
            scores = self.feature_selector(cnn_input_window)

            if original_sensor_padding_mask is not None:
                scores = scores.masked_fill(original_sensor_padding_mask, 0.0)

            top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance_selected = illuminance[batch_indices, top_k_indices, :]
            illuminance = illuminance_selected * top_k_scores.unsqueeze(-1)
            coords = coords[batch_indices, top_k_indices, :]
            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]
            N = self.k

        elif self.scale == 'reductionVariance':
            input_window = illuminance[:, :, :self.projection.in_features]
            variance_scores = torch.var(input_window, dim=2)
            _, top_k_indices = torch.topk(variance_scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance = illuminance[batch_indices, top_k_indices, :]
            coords = coords[batch_indices, top_k_indices, :]
            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k
        sub_windows = illuminance.unfold(2, self.projection.in_features, 1)
        SeqLen = sub_windows.shape[2]
        features = self.projection(sub_windows.permute(0, 2, 1, 3))
        total_pe = torch.zeros_like(features)
        if self.model_mode in ['time_sensor', 'time_sensor_no_sensor_pe', 'time_sensor_wifi_like']:
            dummy_tensor_for_time = torch.zeros(B, SeqLen, C, device=illuminance.device)
            temporal_pe = self.temporal_pe_generator(dummy_tensor_for_time)
            total_pe = total_pe + temporal_pe.unsqueeze(2)

        if self.model_mode == 'time_sensor_wifi_like':
            dummy_tensor_for_sensor = torch.zeros(B, N, C, device=illuminance.device)
            spatial_pe = self.spatial_pe_generator(dummy_tensor_for_sensor)
            total_pe = total_pe + spatial_pe.unsqueeze(1)
            
        elif self.model_mode in ['time_sensor', 'time_sensor_no_time_pe']:
            spatial_pe = self.spatial_pe_generator(coords)
            total_pe = total_pe + spatial_pe.unsqueeze(1)

        if self.args.environment == 'PE' and env_pe is not None:
            total_pe = total_pe + env_pe.unsqueeze(1).unsqueeze(2)
        features_reshaped = features.permute(0, 3, 1, 2)
        pos_reshaped = total_pe.permute(0, 3, 1, 2)
        mask = torch.zeros((B, SeqLen, N), dtype=torch.bool, device=features.device)
        if sensor_padding_mask is not None:
            sensor_mask_expanded = sensor_padding_mask.unsqueeze(1)
            mask = mask | sensor_mask_expanded

        feature_nested_tensor = NestedTensor(features_reshaped, mask)
        attn_mask = None
        if self.scale == 'sparse':
            attn_mask = self._create_sparse_attn_mask(coords, SeqLen, illuminance.device)
        
        return [feature_nested_tensor], [pos_reshaped], attn_mask, predicted_center_coords, selected_sensor_indices


def build_backbone(args):
    if args.model_mode == 'sensor_interpolation':
        print(f"--- Building model with InterpolationBackbone (mode: {args.model_mode}) ---")
        position_embedding = build_position_encoding(args)
        backbone = InterpolationBackbone(
            window_size=args.window_size,
            hidden_dim=args.hidden_dim,
            grid_size=args.grid_size
        )
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
    elif args.model_mode in ['sensor', 'sensor_no_pe']:
        print(f"--- Building model with DirectSensorBackbone (mode: {args.model_mode}) ---")
        model = DirectSensorBackbone(
            window_size=args.window_size,
            hidden_dim=args.hidden_dim,
            args=args,
            model_mode=args.model_mode
        )
    elif args.model_mode == 'time':
        print(f"--- Building model with TimeSequenceBackbone (mode: {args.model_mode}) ---")
        model = TimeSequenceBackbone(
            window_size=args.window_size,
            sub_window_size=args.sub_window_size,
            stride=args.stride,
            num_sensors=args.actual_num_sensors,
            hidden_dim=args.hidden_dim,
            args=args
        )
    elif args.model_mode in ['time_sensor', 'time_sensor_no_time_pe', 'time_sensor_no_sensor_pe', 'time_sensor_no_pe', 'time_sensor_wifi_like']:
        print(f"--- Building model with TimeSensorBackbone (mode: {args.model_mode}, scale: {args.scale}) ---")
        if args.scale == 'sparse':
             print(f"    - Sparse Attention enabled with k={args.k_neighbors} nearest neighbors.")

        model = TimeSensorBackbone(
            sub_window_size=args.sub_window_size,
            num_sensors=args.actual_num_sensors,
            hidden_dim=args.hidden_dim,
            args=args,
            scale=args.scale,
            k=args.k_neighbors,
            model_mode=args.model_mode,
        )
    
    else:
        raise ValueError(f"Unknown model_mode: {args.model_mode}")

    return model