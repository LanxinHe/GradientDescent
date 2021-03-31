import torch
import torch.nn as nn


"""
In this Module, we use Hty and HtH to predict the transform matrix
"""


class ProjectMatrix(nn.Module):
    def __init__(self, tx, dim_z, hidden_size, gru_layers, bi_directional):
        super(ProjectMatrix, self).__init__()
        self.gru = nn.GRU(2*tx+1, hidden_size, num_layers=gru_layers, bidirectional=bi_directional)
        self.sigmoid = nn.Sigmoid()
        # self.dim_z = dim_z

        if bi_directional:
            num_directions = 2
        else:
            num_directions = 1
        self.w_Wz = nn.Parameter(torch.randn([dim_z, num_directions * hidden_size]))
        # self.b_H = nn.Parameter(torch.zeros([2 * tx]))
        self.w_Wx = nn.Parameter(torch.randn([dim_z, num_directions * hidden_size]))
        # self.b_y = nn.Parameter(torch.randn([1]))

    def forward(self, inputs):
        """
        :param inputs: cat(Hty, HtH) in shape of (batch_size, 2tx, 2tx+1)
        :return:
        """
        gru_inputs = inputs.permute(1, 0, 2)
        gru_outputs, _ = self.gru(gru_inputs)   # gru_outputs(2tx, batch_size, hidden_size)
        w_z = torch.matmul(gru_outputs, self.w_Wz.T)  # (2tx, batch_size, dim_z)
        w_z = w_z.permute(1, 2, 0)   # (batch_size, dim_z, 2tx)
        w_x = torch.matmul(gru_outputs, self.w_Wx.T)  # (2tx, batch_size, dim_z)
        w_x = w_x.permute(1, 0, 2)  # (batch_size, 2tx, dim_z)

        return w_z, w_x


class DetModel(nn.Module):
    def __init__(self, tx, rate, dim_z, gru_hidden_size, gru_layers, bi_directional):
        super(DetModel, self).__init__()
        self.tx = tx
        self.rate = rate
        self.project_cal = ProjectMatrix(tx, dim_z, gru_hidden_size, gru_layers, bi_directional)
        self.bm = nn.BatchNorm1d(2 * tx, affine=True)

    def forward(self, inputs, step_size, iterations):
        """
        :param inputs: (y, H); y(batch_size, 2rx) H(batch_size, 2rx, 2tx)
        :return:
        """
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        rnn_inputs = self.bm(torch.cat([HtH, Hty], dim=-1))
        # rnn_inputs = torch.cat([HtH, Hty], dim=-1)
        w_z, w_x = self.project_cal(rnn_inputs)

        x_hat = torch.randint(2 ** self.rate, [batch_size, 2 * self.tx, 1])  # 16QAM
        x_hat = (2 * x_hat - 2 ** self.rate + 1).to(torch.float32)
        z = torch.bmm(w_z, x_hat)   # (batch_size, dim_z, 1)

        z_hat = gradient_descent(Hty, HtH, w_x, step_size=0.000001, iterations=10, z_hat=z)
        x_hat = torch.bmm(w_x, z_hat)

        return x_hat


def gradient_descent(hty, hth, w_x, step_size, iterations, z_hat):
    wxt_hty = torch.bmm(torch.transpose(w_x, dim0=-1, dim1=1), hty)
    whhw = torch.bmm(torch.transpose(w_x, dim0=-1, dim1=1), torch.bmm(hth, w_x))
    for i in range(iterations):
        z_hat = z_hat + 2 * step_size * (wxt_hty - torch.bmm(whhw, z_hat))
    return z_hat
