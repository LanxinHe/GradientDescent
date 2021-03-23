import torch
import torch.nn as nn


"""
In this Module, we use Hty and HtH to predict the fixed delta
"""


class ExtensionL(nn.Module):
    def __init__(self, tx, hidden_size, gru_layers, bi_directional):
        super(ExtensionL, self).__init__()
        self.gru = nn.GRU(2*tx+1, hidden_size, num_layers=gru_layers, bidirectional=bi_directional)
        self.sigmoid = nn.Sigmoid()

        if bi_directional:
            num_directions = 2
        else:
            num_directions = 1
        self.w_H = nn.Parameter(torch.randn([2 * tx, num_directions * hidden_size]))
        self.b_H = nn.Parameter(torch.zeros([2 * tx]))
        self.w_y = nn.Parameter(torch.randn([1, num_directions * hidden_size]))
        self.b_y = nn.Parameter(torch.randn([1]))

    def forward(self, inputs):
        """
        :param inputs: cat(Hty, HtH) in shape of (batch_size, 2tx, 2tx+1)
        :return:
        """
        gru_inputs = inputs.permute(1, 0, 2)
        gru_outputs, _ = self.gru(gru_inputs)   # gru_outputs(2tx, batch_size, hidden_size)
        extension_h = self.sigmoid(torch.matmul(gru_outputs, self.w_H.T) + self.b_H)  # (2tx, batch_size, 2tx)
        extension_h = extension_h.permute(1, 0, 2)   # (batch_size, 2tx, 2tx)
        extension_y = self.sigmoid(torch.matmul(gru_outputs, self.w_y.T) + self.b_y)  # (2tx, batch_size, 1)
        extension_y = extension_y.permute(1, 0, 2)

        return extension_y, extension_h


class DetModel(nn.Module):
    def __init__(self, tx, gru_hidden_size, gru_layers, bi_directional, sigma):
        super(DetModel, self).__init__()
        self.sigma = sigma
        self.extension_cal = ExtensionL(tx, gru_hidden_size, gru_layers, bi_directional)
        self.bm = nn.BatchNorm1d(2 * tx, affine=True)

    def forward(self, inputs):
        """
        :param inputs: (y, H); y(batch_size, 2rx) H(batch_size, 2rx, 2tx)
        :return:
        """
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        rnn_inputs = self.bm(torch.cat([HtH, Hty], dim=-1))

        ext_y, ext_h = self.extension_cal(rnn_inputs)
        ext_y = self.sigma * ext_y  # (batch_size, 2tx, 1)
        ext_h = self.sigma * ext_h  # (batch_size, 2tx, 2tx)

        ext_y = torch.cat([y.unsqueeze(-1), ext_y], dim=1)   # (batch_size, 2rx+2tx, 1)
        ext_h = torch.cat([H, ext_h], dim=1)    # (batch_size, 2rx+2tx, 2tx)

        x_hat = torch.bmm(torch.pinverse(ext_h), ext_y)

        return x_hat
