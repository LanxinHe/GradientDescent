import torch


def loss_calculate(x_oh, label_oh, loss_fn):
    loss = 0.
    for i, prediction in enumerate(x_oh, 0):
        loss += loss_fn(prediction, label_oh) * torch.log(torch.Tensor([i+1]))
    return loss


def last_layer_loss(predictions, labels, loss_fn):
    modulated_labels = (2 * labels - 1).to(torch.float32)
    loss = loss_fn(predictions[-1], modulated_labels)
    return loss


def rnn_loss(predictions, labels, loss_fn):
    """ this loss calculation function is for QPSK."""
    loss = 0.
    modulated_labels = (2 * labels - 1).to(torch.float32)
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(prediction, modulated_labels) * torch.log(torch.Tensor([i+1]))
    return loss


def gpu_loss(predictions, labels, loss_fn):
    """ this loss calculation function is for QPSK in GPU mode"""
    loss = 0.
    modulated_labels = (2 * labels - 1).to(torch.float32)
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(prediction, modulated_labels).cpu() * torch.log(torch.Tensor([i+1]))
    return loss


def gpu_loss_finetune(predictions, labels, loss_fn):
    """ this loss calculation function is for QPSK in GPU mode"""
    loss = 0.
    modulated_labels = (2 * labels - 1).to(torch.float32)
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(prediction, modulated_labels).cpu() * torch.log(torch.Tensor([i+2]))
    return loss


def gpu_loss_16qam(predictions, labels, loss_fn):
    loss = 0.
    modulated_labels = (2 * labels - 3).to(torch.float32)
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(prediction, modulated_labels).cpu() * torch.log(torch.Tensor([i+1]))
    return loss


def common_loss(prediction, labels, rate):
    loss_fn = torch.nn.MSELoss()
    modulated_labels = (2 * labels - 2**rate + 1).to(torch.float32)
    loss = loss_fn(prediction.squeeze(-1), modulated_labels)
    return loss


def ml_loss(predictions, y, h_com):
    """this is the ML(Maximum Likelihood) Loss with log weights:"""
    loss_fn = torch.nn.MSELoss().cuda()
    loss = 0.
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(y, torch.matmul(h_com,
                                        prediction.unsqueeze(-1)).squeeze(-1)).cpu() * torch.log(torch.Tensor([i+2]))
    return loss


def ml_loss_single(prediction, y, h_com):
    """this is the ML(Maximum likelihood) Loss without weights:"""
    loss_fn = torch.nn.MSELoss().cuda()
    loss = loss_fn(y, torch.bmm(h_com, prediction.view(y.shape[0], -1, 1)).squeeze(-1)).cpu()
    return loss


def weighted_mse(predictions, labels, rate):
    loss_fn = torch.nn.MSELoss()
    modulated_labels = (2 * labels - 2**rate + 1).to(torch.float32)
    loss = 0
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(prediction, modulated_labels) * torch.log(torch.Tensor([i+1]))
    return loss