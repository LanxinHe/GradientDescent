import torch
import torch.nn as nn
import sparselinear

import abc
import math
import numpy as np


"""
In this Module, we rewrite DetNet by utilizing RNNCell in PyTorch while only project ONCE at last (MMSE)
use weighted loss function and Sparsely Connected Network
"""


class RecurrentCell(nn.Module):
    def __init__(self, tx, hidden_size):
        super(RecurrentCell, self).__init__()
        weight_sparsity = 0.4
        self.linear_h = nn.Linear(hidden_size+2*tx, hidden_size)
        normalizeSparseWeights(self.linear_h, weight_sparsity)
        self.linear_h = SparseWeights(self.linear_h, weight_sparsity)

        self.k_winner = sparselinear.ActivationSparsity(act_sparsity=0.4)
        self.linear_x = nn.Linear(hidden_size, 2*tx)
        normalizeSparseWeights(self.linear_x, weight_sparsity)
        self.linear_x = SparseWeights(self.linear_x, weight_sparsity)

    def forward(self, z, h_pre):
        """
        :param z: in shape of (batch_size, 2tx)
        :param h_pre: the previous hidden state in shape of (batch_size, hidden_size)
        :return:
        """
        x = z
        # h (batch_size, hidden_size)
        cat = torch.cat([x, h_pre], dim=-1)
        h = self.k_winner(self.linear_h(cat))
        x = self.linear_x(h)

        return x, h


class DetModel(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times):
        super(DetModel, self).__init__()
        self.r_cell = RecurrentCell(tx, rnn_hidden_size)
        self.project_times = project_times

    def forward(self, inputs, x, h, step_size, iterations):
        """
        :param inputs: (y, H); y(batch_size, 2rx+2tx) H(batch_size, 2rx+2tx, 2tx)
        :return:
        """
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        outputs = []
        for p in range(self.project_times):
            for i in range(iterations):
                x = gradient_descent(Hty, HtH, step_size, x)
            x, h = self.r_cell(x.squeeze(-1), h)
            outputs += [x]
            x = x.unsqueeze(-1)
        x = x.squeeze(-1)
        return x, h, outputs


def gradient_descent(hty, hth, step_size, x_pre):
    z = x_pre + 2 * step_size * (hty - torch.bmm(hth, x_pre))
    return z


def rezeroWeights(m):
    """
    Function used to update the weights after each epoch.
    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(rezeroWeights)``
    :param m: SparseWeightsBase module
    """
    if isinstance(m, SparseWeightsBase):
        if m.training:
          m.rezeroWeights()


def normalizeSparseWeights(m, weightSparsity):
    """
    Initialize the weights using kaiming_uniform initialization normalized to
    the number of non-zeros in the layer instead of the whole input size.
    Similar to torch.nn.Linear.reset_parameters() but applying weight sparsity
    to the input size
    """

    _, inputSize = m.weight.shape
    fan = int(inputSize * weightSparsity)
    gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
    std = gain / np.math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    nn.init.uniform_(m.weight, -bound, bound)
    if m.bias is not None:
      bound = 1 / math.sqrt(fan)
      nn.init.uniform_(m.bias, -bound, bound)


class SparseWeightsBase(nn.Module):
    """
    Base class for the all Sparse Weights modules
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, module, weightSparsity):
        """
        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(SparseWeightsBase, self).__init__()
        assert 0 < weightSparsity < 1

        self.module = module
        self.weightSparsity = weightSparsity
        self.register_buffer("zeroWts", self.computeIndices())
        self.rezeroWeights()

    def forward(self, x):
        if self.training:
          self.rezeroWeights()
        return self.module.forward(x)

    @abc.abstractmethod
    def computeIndices(self):
        """
        For each unit, decide which weights are going to be zero
        :return: tensor indices for all non-zero weights. See :meth:`rezeroWeights`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rezeroWeights(self):
        """
        Set the previously selected weights to zero. See :meth:`computeIndices`
        """
        raise NotImplementedError


class SparseWeights(SparseWeightsBase):
    def __init__(self, module, weightSparsity):
        """
        Enforce weight sparsity on linear module during training.
        Sample usage:
          model = nn.Linear(784, 10)
          model = SparseWeights(model, 0.4)
        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(SparseWeights, self).__init__(module, weightSparsity)

    def computeIndices(self):
        # For each unit, decide which weights are going to be zero
        outputSize, inputSize = self.module.weight.shape
        numZeros = int(round((1.0 - self.weightSparsity) * inputSize))

        outputIndices = np.arange(outputSize)
        inputIndices = np.array([np.random.permutation(inputSize)[:numZeros]
                                 for _ in outputIndices], dtype=np.long)

        # Create tensor indices for all non-zero weights
        zeroIndices = np.empty((outputSize, numZeros, 2), dtype=np.long)
        zeroIndices[:, :, 0] = outputIndices[:, None]
        zeroIndices[:, :, 1] = inputIndices
        zeroIndices = zeroIndices.reshape(-1, 2)
        return torch.from_numpy(zeroIndices.transpose())

    def rezeroWeights(self):
        zeroIdx = (self.zeroWts[0].to(torch.long), self.zeroWts[1].to(torch.long))
        self.module.weight.data[zeroIdx] = 0.0
