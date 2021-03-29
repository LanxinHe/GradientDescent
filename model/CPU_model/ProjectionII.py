import torch
import torch.nn as nn


"""
In this Module, we use Hty and HtH to predict the transform matrix
"""


class ProjectMatrix(nn.Module):
    def __init__(self, tx, rx, dim_z):
        super(ProjectMatrix, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.w1 = nn.Parameter(torch.randn([dim_z, 2*rx]))
        self.w2 = nn.Parameter(torch.randn([2*tx, dim_z]))

    def forward(self, h_matrix):
        """
        :param h_matrix: channel matrix, in shape of (batch_size, 2rx, 2tx)
        :return:
        """
        w_z = self.sigmoid(torch.matmul(self.w1, h_matrix))
        w_x = self.sigmoid(torch.matmul(h_matrix, self.w2))

        return w_z, w_x


class DetModel(nn.Module):
    def __init__(self, tx, rx, rate, dim_z,):
        super(DetModel, self).__init__()
        self.tx = tx
        self.rate = rate
        self.project_cal = ProjectMatrix(tx, rx, dim_z)
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

        # rnn_inputs = self.bm(torch.cat([HtH, Hty], dim=-1))
        # rnn_inputs = torch.cat([HtH, Hty], dim=-1)
        w_z, w_x = self.project_cal(H)

        x_hat = torch.randint(2 ** self.rate, [batch_size, 2 * self.tx, 1])  # 16QAM
        x_hat = (2 * x_hat - 2 ** self.rate + 1).to(torch.float32)
        z = torch.bmm(w_z, x_hat)   # (batch_size, dim_z, 1)

        z_hat = gradient_descent(Hty, HtH, w_x, step_size, iterations, z)
        x_hat = torch.bmm(w_x, z_hat)

        return x_hat


def gradient_descent(hty, hth, w_x, step_size, iterations, z_hat):
    wxt_hty = torch.bmm(torch.transpose(w_x, dim0=-1, dim1=1), hty)
    whhw = torch.bmm(torch.transpose(w_x, dim0=-1, dim1=1), torch.bmm(hth, w_x))
    for i in range(iterations):
        z_hat = z_hat + 2 * step_size * (wxt_hty - torch.bmm(whhw, z_hat))
    return z_hat
