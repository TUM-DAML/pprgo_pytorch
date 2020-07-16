import math
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        value_dropped = F.dropout(input.storage.value(), self.p, self.training)
        return torch_sparse.SparseTensor(
                row=input.storage.row(), rowptr=input.storage.rowptr(), col=input.storage.col(),
                value=value_dropped, sparse_sizes=input.sparse_sizes(), is_sorted=True)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            res = input.matmul(self.weight)
            if self.bias:
                res += self.bias[None, :]
        else:
            if self.bias:
                res = torch.addmm(self.bias, input, self.weight)
            else:
                res = input.matmul(self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def matrix_to_torch(X):
    if sp.issparse(X):
        return torch_sparse.SparseTensor.from_scipy(X)
    else:
        return torch.FloatTensor(X)
