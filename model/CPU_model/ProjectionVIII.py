import torch
import torch.nn as nn


"""
In this Module, we rewrite DetNet by utilizing RNNCell in PyTorch while only project ONCE at last (MMSE)
use weighted loss function and Leaky RELU
"""


class RecurrentCell(nn.Module):
    def __init__(self, tx, hidden_size):
        super(RecurrentCell, self).__init__()
        sqrt_k = torch.sqrt(torch.Tensor([1/hidden_size]))
        self.w_ih = nn.Parameter(sqrt_k * (2*torch.rand([hidden_size, 2*tx])-1))
        self.w_hh = nn.Parameter(sqrt_k * (2*torch.rand([hidden_size, hidden_size])-1))
        self.b_ih = nn.Parameter(sqrt_k * (2 * torch.rand([hidden_size]) - 1))
        self.b_hh = nn.Parameter(sqrt_k * (2 * torch.rand([hidden_size]) - 1))
        self.leaky_relu = nn.LeakyReLU()

        self.w_x = nn.Parameter(torch.randn([2*tx, hidden_size]))
        self.b_x = nn.Parameter(torch.randn([2*tx]))

    def forward(self, z, h_pre):
        """
        :param z: in shape of (batch_size, 2tx)
        :param h_pre: the previous hidden state in shape of (batch_size, hidden_size)
        :return:
        """
        x = z
        # h (batch_size, hidden_size)
        h = self.leaky_relu(torch.matmul(x, self.w_ih.T) + self.b_ih + torch.matmul(h_pre, self.w_hh.T) + self.b_hh)
        x = torch.matmul(h, self.w_x.T) + self.b_x

        return x, h


class DetModel(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times):
        super(DetModel, self).__init__()
        self.r_cell = RecurrentCell(tx, rnn_hidden_size)
        self.project_times = project_times

        # self.bm = nn.BatchNorm1d(2 * tx, affine=True)

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
