"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

def box_cxcywhd_to_xyzxyz(x):
    """ (cx,cy,cz,w,h,d) -> (x1,y1,z1,x2,y2,z2) """
    cx, cy, cz, w, h, d = x.unbind(-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cz - 0.5 * d,
         cx + 0.5 * w, cy + 0.5 * h, cz + 0.5 * d]
    return torch.stack(b, dim=-1)

class EnvQueryCNN(nn.Module):
    """
    環境ベクトル [B, N] を受け取り、クエリ用のPE [B, C] を出力するCNN (MLP)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args

        self.env_query_cnn = None
        self.env_seq_proj = None
        self.env_seq_pe = None

        if self.args is not None:
            if self.args.environment == 'query':
                self.env_query_cnn = EnvQueryCNN(
                    input_dim=self.args.actual_num_sensors,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim
                )
            elif self.args.environment == 'sequence':
                self.env_seq_proj = EnvQueryCNN(
                    input_dim=self.args.actual_num_sensors,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim
                )
                self.env_seq_pe = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1), requires_grad=False)
                nn.init.xavier_uniform_(self.env_seq_pe)

    def forward(self, samples: NestedTensor):
        backbone_output = self.backbone(samples)
        predicted_center_coords = None
        selected_sensor_indices = None

        if len(backbone_output) == 5:
            features, pos, attn_mask, predicted_center_coords, selected_sensor_indices = backbone_output
        elif len(backbone_output) == 4:
            features, pos, attn_mask, predicted_center_coords = backbone_output
        elif len(backbone_output) == 3:
            features, pos, attn_mask = backbone_output
        else:
            features, pos = backbone_output
            attn_mask = None

        src, mask = features[-1].decompose()
        assert mask is not None

        src = self.input_proj(src)
        
        p = pos[-1]
        
        if self.env_seq_proj is not None and 'env_vector' in samples:
            env_vector = samples['env_vector']
            B = env_vector.shape[0]
            
            env_token = self.env_seq_proj(env_vector).unsqueeze(2) 
            
            env_pe = self.env_seq_pe.squeeze(-1).repeat(B, 1, 1)

            src_flat = src.flatten(2)
            mask_flat = mask.flatten(1)
            p_flat = p.flatten(2)
            
            src_concat = torch.cat([env_token, src_flat], dim=2)
            mask_concat = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), mask_flat], dim=1)
            p_concat = torch.cat([env_pe, p_flat], dim=2)
            
            src = src_concat.unsqueeze(-1)
            mask = mask_concat.unsqueeze(-1)
            pos_tensor = p_concat.unsqueeze(-1)

            if attn_mask is not None:
                L = attn_mask.shape[0]
                new_attn_mask = torch.zeros((L + 1, L + 1), dtype=torch.bool, device=attn_mask.device)
                new_attn_mask[1:, 1:] = attn_mask
                attn_mask = new_attn_mask
        else:
            pos_tensor = p

        hs = self.transformer(src, mask, self.query_embed.weight, pos_tensor, attn_mask=attn_mask)[0]
        
        if self.env_query_cnn is not None:
            if 'env_vector' in samples:
                env_vector = samples['env_vector']
                env_query_pe = self.env_query_cnn(env_vector)
                hs = hs + env_query_pe.unsqueeze(0).unsqueeze(2)
            else:
                print("Warning: 'env_vector' not found in samples during DETR forward. Skipping query addition.")

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        if predicted_center_coords is not None:
            out['pred_center_coords'] = predicted_center_coords
        if selected_sensor_indices is not None:
            out['selected_sensor_indices'] = selected_sensor_indices
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_center_coords(self, outputs, targets, indices, num_boxes):
        """Compute the loss for the center coordinate prediction (for reduction mode)."""
        if 'pred_center_coords' not in outputs or outputs['pred_center_coords'] is None:
            return {}

        valid_targets = [t for t in targets if 'center_coords' in t]
        if not valid_targets:
            return {}
            
        src_coords = outputs['pred_center_coords']
        
        batch_indices = [i for i, t in enumerate(targets) if 'center_coords' in t]
        
        src_coords_filtered = src_coords[batch_indices]
        target_coords = torch.stack([t['center_coords'] for t in valid_targets])

        if len(src_coords_filtered) == 0:
            return {}

        loss = F.mse_loss(src_coords_filtered, target_coords)
        return {'loss_center': loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        if 'loss_giou' in self.weight_dict:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'center': self.loss_center_coords,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def __init__(self, num_classes):
        super().__init__()
        self.no_object_class_idx = num_classes
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 3

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1) 
        keep = labels != self.no_object_class_idx

        boxes = box_cxcywhd_to_xyzxyz(out_bbox)
        
        img_h, img_w, img_d = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_d, img_w, img_h, img_d], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b, k in zip(scores, labels, boxes, keep):
            results.append({'scores': s[k], 'labels': l[k], 'boxes': b[k]})
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )
    args.set_cost_giou = 0
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    if args.scale == 'reductionCNNmse':
        weight_dict['loss_center'] = args.center_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.scale == 'reductionCNNmse':
        losses.append('center')

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_classes=num_classes)}

    return model, criterion, postprocessors
