import torch
import torch.nn.functional as F


def online_head_classification(query, key, anchor_len, mid_len, sample_divisor=3):
    """Estimate per-head attention usage over (anchor, mid, current) token chunks."""
    _, q_len, _, head_dim = query.shape
    num_sampled_rows = max(1, q_len // sample_divisor)
    sampled_rows = torch.randint(low=0, high=q_len, size=(num_sampled_rows,), device=query.device)
    sampled_q = query[:, sampled_rows].transpose(1, 2)
    key = key.transpose(1, 2)
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (head_dim ** 0.5)
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)

    current_len = q_len
    anchor_agg = sampled_attn_weights[:, :, :, :anchor_len].sum(dim=-1).mean(dim=-1)
    mid_agg = sampled_attn_weights[:, :, :, anchor_len:anchor_len + mid_len].sum(dim=-1).mean(dim=-1)
    current_agg = sampled_attn_weights[:, :, :, -current_len:].sum(dim=-1).mean(dim=-1)
    return torch.stack([anchor_agg, mid_agg, current_agg])


def dynamic_head_programming(probs, num_dummy=180):
    """Split heads into sink/current-heavy and mid-heavy groups with a global budget."""
    num_layer, num_head, _ = probs.shape
    p_anchor = probs[:, :, 0].reshape(-1)
    p_mid = probs[:, :, 1].reshape(-1)
    p_anchor_norm = p_anchor / p_anchor.sum().clamp_min(1e-6)
    p_mid_norm = p_mid / p_mid.sum().clamp_min(1e-6)

    # Heads with low contribution to both groups become dummy heads.
    cost = torch.maximum(p_anchor_norm, p_mid_norm)
    sorted_indices = torch.argsort(cost)
    dummy_indices_flat = sorted_indices[:num_dummy]

    assignment = torch.zeros(num_layer * num_head, dtype=torch.long, device=probs.device)
    assignment[dummy_indices_flat] = 2
    remaining_indices = torch.nonzero(assignment != 2, as_tuple=True)[0]
    for idx in remaining_indices:
        assignment[idx] = 1 if p_anchor_norm[idx] < p_mid_norm[idx] else 0

    assignment = assignment.reshape(num_layer, num_head)
    group_sink = {}
    group_mid = {}
    for layer_idx in range(num_layer):
        sink_idx = (assignment[layer_idx] != 1).nonzero(as_tuple=True)[0].tolist()
        mid_idx = (assignment[layer_idx] == 1).nonzero(as_tuple=True)[0].tolist()
        group_sink[layer_idx] = sink_idx
        group_mid[layer_idx] = mid_idx
    return group_sink, group_mid


def heterogeneous_memory_allocation(global_kv_cache, num_dummy=180):
    global_frame_attn_score = torch.stack(
        [layer_info['frame_attn_score'][:, 0] for layer_info in global_kv_cache]
    ).transpose(1, 2)
    global_group_sink, global_group_mid = dynamic_head_programming(global_frame_attn_score, num_dummy)
    for layer_idx, cur_cache in enumerate(global_kv_cache):
        cur_cache['headgroup_sink'] = global_group_sink[layer_idx]
        cur_cache['headgroup_mid'] = global_group_mid[layer_idx]

