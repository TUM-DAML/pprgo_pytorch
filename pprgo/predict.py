import time
import numpy as np
import torch

from .pytorch_utils import matrix_to_torch


def get_local_logits(model, attr_matrix, batch_size=10000):
    device = next(model.parameters()).device

    nnodes = attr_matrix.shape[0]
    logits = []
    with torch.set_grad_enabled(False):
        for i in range(0, nnodes, batch_size):
            batch_attr = matrix_to_torch(attr_matrix[i:i + batch_size]).to(device)
            logits.append(model(batch_attr).to('cpu').numpy())
    logits = np.row_stack(logits)
    return logits


def predict(model, adj_matrix, attr_matrix, alpha,
            nprop=2, inf_fraction=1.0, ppr_normalization='sym', batch_size_logits=10000):

    model.eval()

    start = time.time()
    if inf_fraction < 1.0:
        idx_sub = np.random.choice(adj_matrix.shape[0], int(inf_fraction * adj_matrix.shape[0]), replace=False)
        idx_sub.sort()
        attr_sub = attr_matrix[idx_sub]
        logits_sub = get_local_logits(model.mlp, attr_sub, batch_size_logits)
        local_logits = np.zeros([adj_matrix.shape[0], logits_sub.shape[1]], dtype=np.float32)
        local_logits[idx_sub] = logits_sub
    else:
        local_logits = get_local_logits(model.mlp, attr_matrix, batch_size_logits)
    time_logits = time.time() - start

    start = time.time()
    row, col = adj_matrix.nonzero()
    logits = local_logits.copy()

    if ppr_normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
        for _ in range(nprop):
            logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'col':
        deg_col = adj_matrix.sum(0).A1
        deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
        for _ in range(nprop):
            logits = (1 - alpha) * (adj_matrix @ (deg_col_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'row':
        deg_row = adj_matrix.sum(1).A1
        deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            logits = deg_row_inv_alpha[:, None] * (adj_matrix @ logits) + alpha * local_logits
    else:
        raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
    predictions = logits.argmax(1)
    time_propagation = time.time() - start

    return predictions, time_logits, time_propagation
