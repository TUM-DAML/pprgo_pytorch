import torch

from .pytorch_utils import matrix_to_torch


# TODO: Caching, prefetching (?)
# def feed_for_batch(self, attr_matrix, ppr_matrix, labels, key=None):
#     if key is None:
#         return self.gen_feed(attr_matrix, ppr_matrix, labels)
#     else:
#         if key in self.cached:
#             return self.cached[key]
#         else:
#             feed = self.gen_feed(attr_matrix, ppr_matrix, labels)
#             self.cached[key] = feed
#             return feed

class PPRDataset(torch.utils.data.Dataset):
    def __init__(self, attr_matrix_all, ppr_matrix, indices, labels_all=None):
        self.attr_matrix_all = attr_matrix_all
        self.ppr_matrix = ppr_matrix
        self.indices = indices
        self.labels_all = torch.tensor(labels_all)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # idx is a list of indices
        idx = sorted(idx)
        ppr_matrix = self.ppr_matrix[idx]
        source_idx, neighbor_idx = ppr_matrix.nonzero()
        ppr_scores = ppr_matrix.data

        attr_matrix = matrix_to_torch(self.attr_matrix_all[neighbor_idx])
        ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)
        source_idx = torch.tensor(source_idx, dtype=torch.long)

        if self.labels_all is None:
            labels = None
        else:
            labels = self.labels_all[self.indices[idx]]
        return (attr_matrix, ppr_scores, source_idx), labels
