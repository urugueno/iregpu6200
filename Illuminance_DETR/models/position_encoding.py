import math
import torch
from torch import nn

from util.misc import NestedTensor

class PositionEmbeddingSine1D(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        b, c, h, w = x.shape
        assert c == self.num_pos_feats, \
            f"The channel dimension {c} of the input tensor must match the num_pos_feats {self.num_pos_feats}"

        seq_len = h * w
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)

        dim_t = torch.arange(0, self.num_pos_feats, 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (dim_t / self.num_pos_feats)

        pe = torch.zeros(seq_len, self.num_pos_feats, device=x.device)
        pe[:, 0::2] = torch.sin(position / dim_t)
        pe[:, 1::2] = torch.cos(position / dim_t)

        pos = pe.T.view(1, c, h, w)
        pos = pos.repeat(b, 1, 1, 1)
        
        return pos

class PositionEmbeddingFromCoords(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, coords: torch.Tensor):
        coords = coords * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=coords.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = coords[..., 0].unsqueeze(-1) / dim_t
        pos_y = coords[..., 1].unsqueeze(-1) / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2)

class PositionEmbeddingLearned1D(nn.Module):
    def __init__(self, num_embeddings, num_pos_feats=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight)

    def forward(self, sensor_ids):
        return self.embedding(sensor_ids)

class PositionEmbeddingTime1D(nn.Module):
    def __init__(self, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        b, seq_len, c = x.shape
        assert c == self.num_pos_feats, \
            f"The channel dimension {c} of the input tensor must match num_pos_feats {self.num_pos_feats}"

        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, c, 2, dtype=torch.float32, device=x.device) * \
                             (-math.log(self.temperature) / c))
        
        pe = torch.zeros(seq_len, c, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0).repeat(b, 1, 1)