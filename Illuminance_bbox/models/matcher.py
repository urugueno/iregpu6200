import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_coord: float = 1, task: str = 'bbox'):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        self.task = task
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]

        if self.task == 'pose':
            out_coord = outputs["pred_keypoints"].flatten(0, 1).flatten(1)
            tgt_coord = torch.cat([v["keypoints"] for v in targets]).flatten(1)
            sizes = [len(v["keypoints"]) for v in targets]
        else:
            out_coord = outputs["pred_boxes"].flatten(0, 1)
            tgt_coord = torch.cat([v["boxes"] for v in targets])
            sizes = [len(v["boxes"]) for v in targets]

        cost_coord = torch.cdist(out_coord, tgt_coord, p=1)

        C = self.cost_coord * cost_coord + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        if torch.isnan(C).any() or torch.isinf(C).any():
            C = torch.nan_to_num(C, nan=1e5, posinf=1e5, neginf=-1e5)

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    task = args.task
    cost_class = args.set_cost_class
    
    if task == 'pose':
        cost_coord = args.keypoint_loss_coef
    else:
        cost_coord = args.set_cost_bbox

    return HungarianMatcher(cost_class=cost_class, cost_coord=cost_coord, task=task)