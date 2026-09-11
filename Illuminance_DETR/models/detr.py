import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, accuracy)
from util.calculate import box_cxcywhd_to_xyzxyz

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

class EQCNN(nn.Module):
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
        
        self.task = args.task
        if self.task == 'pose':
            self.coord_embed = MLP(hidden_dim, hidden_dim, 42, 3)
        else:
            self.coord_embed = MLP(hidden_dim, hidden_dim, 6, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args

        self.env_query_cnn = None
        if self.args is not None:
            if self.args.environment in ('EQ', 'PE_query'):
                self.env_query_cnn = EQCNN(
                    input_dim=self.args.actual_num_sensors,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim
                )

    def forward(self, samples: NestedTensor):
        backbone_output = self.backbone(samples)
        selected_sensor_indices = None

        if len(backbone_output) == 4:
            features, pos, attn_mask, selected_sensor_indices = backbone_output
        elif len(backbone_output) == 3:
            features, pos, attn_mask = backbone_output
        else:
            features, pos = backbone_output
            attn_mask = None

        src, mask = features[-1].decompose()
        assert mask is not None

        query_embed = self.query_embed.weight
        if self.env_query_cnn is not None:
            env_vector = samples['env_vector']
            env_query_pe = self.env_query_cnn(env_vector)  # [B, hidden_dim]
            # q'_i = q_i + psi(e): per-sample env vector added to the Object
            # Query fed into the decoder, not to the decoder's final output.
            query_embed = query_embed.unsqueeze(1) + env_query_pe.unsqueeze(0)  # [N, B, hidden_dim]

        hs = self.transformer(self.input_proj(src), mask, query_embed, pos[-1], attn_mask=attn_mask)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.coord_embed(hs).sigmoid()
        
        if self.task == 'pose':
            outputs_class = torch.nan_to_num(outputs_class, nan=0.0, posinf=1e4, neginf=-1e4)
            outputs_coord = torch.nan_to_num(outputs_coord, nan=0.0, posinf=1.0, neginf=0.0)

        out_coord_key = 'pred_keypoints' if self.task == 'pose' else 'pred_boxes'
        out = {'pred_logits': outputs_class[-1], out_coord_key: outputs_coord[-1]}

        if selected_sensor_indices is not None:
            out['selected_sensor_indices'] = selected_sensor_indices
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, out_coord_key)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, coord_key):
        return [{'pred_logits': a, coord_key: b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
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
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx]
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_keypoints = target_keypoints.flatten(1)
        loss_keypoint = F.l1_loss(src_keypoints, target_keypoints, reduction='none')
        return {'loss_keypoint': loss_keypoint.sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        return {'loss_bbox': loss_bbox.sum() / num_boxes}
    
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
            'keypoints': self.loss_keypoints,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

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
    def __init__(self, num_classes, task='bbox'):
        super().__init__()
        self.no_object_class_idx = num_classes
        self.task = task
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits = outputs['pred_logits']
        assert len(out_logits) == len(target_sizes)
        
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1) 
        keep = labels != self.no_object_class_idx
        img_h, img_w, img_d = target_sizes.unbind(1)
        
        results = []
        
        if self.task == 'pose':
            out_keypoints = outputs['pred_keypoints']
            keypoints = out_keypoints.reshape(out_keypoints.shape[0], out_keypoints.shape[1], 14, 3)
            scale_fct = torch.stack([img_w, img_h, img_d], dim=1).view(-1, 1, 1, 3)
            keypoints = keypoints * scale_fct
            for s, l, kpts, k in zip(scores, labels, keypoints, keep):
                results.append({'scores': s[k], 'labels': l[k], 'keypoints': kpts[k]})
                
        else:
            out_bbox = outputs['pred_boxes']
            boxes = box_cxcywhd_to_xyzxyz(out_bbox)
            scale_fct = torch.stack([img_w, img_h, img_d, img_w, img_h, img_d], dim=1)
            boxes = boxes * scale_fct[:, None, :]
            for s, l, b, k in zip(scores, labels, boxes, keep):
                results.append({'scores': s[k], 'labels': l[k], 'boxes': b[k]})
                
        return results
    
class MLP(nn.Module):
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
    task = args.task
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETR(backbone, transformer, num_classes=num_classes, num_queries=args.num_queries, aux_loss=args.aux_loss, args=args)
    matcher = build_matcher(args)
    
    if task == 'pose':
        weight_dict = {'loss_ce': 1, 'loss_keypoint': args.keypoint_loss_coef}
        losses = ['labels', 'keypoints', 'cardinality']
    else:
        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        losses = ['labels', 'boxes', 'cardinality']

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_classes=num_classes, task=task)}

    return model, criterion, postprocessors