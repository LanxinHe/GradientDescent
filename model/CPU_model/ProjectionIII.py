import torch
import torch.nn as nn


"""
In this Module, we use Hty and HtH to predict the transform matrix
"""


class RecurrentCell(nn.Module):
    def __init__(self, tx, hidden_size):
        super(RecurrentCell, self).__init__()
        self.rnn = nn.RNNCell(2*tx, hidden_size, nonlinearity='relu')
        self.sigmoid = nn.Sigmoid()

        self.w_x = nn.Parameter(torch.randn([2*tx, hidden_size]))
        self.b_x = nn.Parameter(torch.randn([2*tx]))

    def forward(self, z, h_pre):
        """
        :param z: in shape of (batch_size, 2tx)
        :param h_pre: the previous hidden state
        :return:
        """
        h = self.rnn(z, h_pre)  # h (batch_size, hidden_size)
        x = torch.matmul(h, self.w_x.T) + self.b_x

        return x, h


class DetModel(nn.Module):
    def __init__(self, tx, rate, rnn_hidden_size):
        super(DetModel, self).__init__()
        self.tx = tx
        self.rate = rate
        self.r_cell = RecurrentCell(tx, rnn_hidden_size)
        self.hidden_size = rnn_hidden_size
        # self.bm = nn.BatchNorm1d(2 * tx, affine=True)

    def forward(self, inputs, step_size, iterations):
        """
        :param inputs: (y, H); y(batch_size, 2rx) H(batch_size, 2rx, 2tx)
        :return:
        """
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        x = torch.randint(2 ** self.rate, [batch_size, 2 * self.tx, 1])  # 16QAM
        x = (2 * x - 2 ** self.rate + 1).to(torch.float32)
        h = torch.zeros([batch_size, self.hidden_size])

        for i in range(iterations):
            x, h = self.gradient_descent(Hty, HtH, step_size, x, h)

        return x, h

    def gradient_descent(self, hty, hth, step_size, x_pre, h_pre):
        z = x_pre + 2 * step_size * (hty - torch.bmm(hth, x_pre))
        z = z.squeeze(-1)
        x, h = self.r_cell(z, h_pre)
        x = x.unsqueeze(-1)
        return x, h
