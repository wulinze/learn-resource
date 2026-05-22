import torch


def group_count_ref(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    out = torch.zeros((num_groups,), dtype=torch.int32, device=group_idx.device)
    valid = group_idx >= 0
    if valid.any():
        values = group_idx[valid].to(torch.int64)
        ones = torch.ones_like(values, dtype=torch.int32)
        out.scatter_add_(0, values, ones)
    return out

