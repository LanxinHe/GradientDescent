from functions import data_preparation
from functions.test_functions import gray_ber
from functions.loss_cal import weighted_mse
from model.CPU_model import ProjectionX
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# --------------------------------------------- Dataset ------------------------------------------------------
class DetDataset(Dataset):
    def __init__(self, y, h_com, data_real, data_imag, transform=None):
        self.y = y
        self.h_com = h_com
        self.label = torch.cat([torch.from_numpy(data_real.T),
                                torch.from_numpy(data_imag.T)],
                               dim=1).long()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'y': torch.from_numpy(self.y[idx, :]).to(torch.float32),
            'h_com': torch.from_numpy(self.h_com[idx, :, :]).to(torch.float32),
            'label': self.label[idx, :],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    TX = 16
    RX = 16
    N_TRAIN = 100
    N_TEST = 1000
    TRAIN_SPLIT = 0.9
    RATE = 2
    EBN0_TRAIN = 10
    LENGTH = 2 ** RATE
    BATCH_SIZE = 20
    EPOCHS = 100
    PROJECT_TIMES = 5
    RNN_HIDDEN_SIZE = 8 * TX
    STEP_SIZE = 0.012
    ITERATIONS = 8

    train_y, train_h_com, train_Data_real, train_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_y, train_h_com, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # ------------------------------------- Establish Network ----------------------------------------------
    # PATH = '../../pretrained_projectionX_Extension/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f' % (TX, RX, RATE,
    #                                                                                                         EBN0_TRAIN,
    #                                                                                                         ITERATIONS,
    #                                                                                                         PROJECT_TIMES,
    #                                                                                                         BATCH_SIZE,
    #                                                                                                         RNN_HIDDEN_SIZE,
    #                                                                                                         STEP_SIZE)
    PATH = '../../pretrained_projectionX_Extension/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/extension3' % (TX, RX, RATE,
                                                                                                            EBN0_TRAIN,
                                                                                                            ITERATIONS,
                                                                                                            PROJECT_TIMES,
                                                                                                            BATCH_SIZE,
                                                                                                            RNN_HIDDEN_SIZE,
                                                                                                            STEP_SIZE)
    prenet1 = ProjectionX.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES)
    prenet1.load_state_dict(torch.load(PATH + str('/model_part1.pt')))
    prenet2 = ProjectionX.DetModel(TX, RNN_HIDDEN_SIZE, 3)
    prenet2.load_state_dict(torch.load(PATH + str('/model_part2.pt')))

    # --------------------------------------------------- Test ---------------------------------------------------------
    with torch.no_grad():
        prenet1.eval()
        prenet2.eval()
        test_loss = 0.0
        test_steps = 0
        predictions = []
        for i, data in enumerate(test_loader, 0):
            y, h_com, label = data['y'], data['h_com'], data['label']
            inputs = (y, h_com)

            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs1 = prenet1(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            x = x.unsqueeze(-1)
            x, h, outputs2 = prenet2(inputs, x, h, STEP_SIZE, ITERATIONS)
            outputs = outputs1 + outputs2
            # x, h, outputs = detnet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            loss = weighted_mse(outputs, label, RATE)
            predictions += [x]
            test_loss += loss.numpy()
            test_steps += 1
        print('test loss: %.3f' % (test_loss / test_steps))
        predictions = torch.cat(predictions).cpu().numpy()
        # predictions = ((predictions + LENGTH - 1)/2).round()
        # ber = gray_ber(predictions, test_Data_real, test_Data_imag, rate=RATE)
    # ------------------------------------------------ Weights Plot ----------------------------------------------------
    style_size = 0.1
    points = outputs2[0].numpy()
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.figure()
    plt.scatter(real_part, imag_part, style_size)

    points = outputs2[1].numpy()
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.scatter(real_part, imag_part, style_size)

    points = predictions
    # points = outputs2[2].numpy()
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.scatter(real_part, imag_part, style_size)

    # points = outputs[3].numpy()
    # real_part = points[:, :TX]
    # imag_part = points[:, TX:]
    # plt.scatter(real_part, imag_part, style_size)

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.xticks([-3, -1, 1, 3])
    plt.yticks([-3, -1, 1, 3])

    plt.legend(['projection', 'projection 2', 'projection 3', 'projection 4'], loc=1)
    plt.title('Projection method for 16QAM')
    plt.xlabel('Re', loc='right', fontdict={'fontsize': 'x-large', 'fontweight': 'bold'})
    plt.ylabel('Im', loc='bottom', fontdict={'fontsize': 'x-large', 'fontweight': 'bold', 'rotation': 'horizontal'})


    # --------------------------------------- Save Model & Data --------------------------------------------------------
    PATH = '../../pretrained_projectionX/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f' % (TX, RX, RATE,
                                                                                                            EBN0_TRAIN,
                                                                                                            ITERATIONS,
                                                                                                            PROJECT_TIMES,
                                                                                                            BATCH_SIZE,
                                                                                                            RNN_HIDDEN_SIZE,
                                                                                                            STEP_SIZE)
    os.makedirs(PATH)
    data_ber = pd.DataFrame(BER, columns=['BER'])
    data_ber.to_csv(PATH+str('/ber.csv'))
    torch.save(detnet.state_dict(), PATH+str('/model.pt'))
    # use the following line to load model
    detnet.load_state_dict(torch.load(PATH + str('/model.pt')))
