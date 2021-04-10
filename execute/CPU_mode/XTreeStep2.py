from functions import data_preparation
from functions.test_functions import gray_ber
from functions.loss_cal import tree_step1, tree_step2
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
    N_TRAIN = 100000
    N_TEST = 2000
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
    PATH = '../../pretrained_projectionXTree/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f' % (TX, RX, RATE,
                                                                                                            EBN0_TRAIN,
                                                                                                            ITERATIONS,
                                                                                                            PROJECT_TIMES,
                                                                                                            BATCH_SIZE,
                                                                                                            RNN_HIDDEN_SIZE,
                                                                                                            STEP_SIZE)
    prenet1 = ProjectionX.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES)
    prenet1.load_state_dict(torch.load(PATH + str('/model_part1.pt')))

    detnet = ProjectionX.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES)
    optim_det = torch.optim.Adam(detnet.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim_det, step_size=10, gamma=0.2)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_det, [10, 20, 35, 50, 70, 90], 0.1)

    # ------------------------------------- 16QAM Train (Step2)------------------------------------------------------
    prenet1.requires_grad_(False)
    print('Begin Training:')
    for epoch in range(50):
        running_loss = 0.0
        detnet.train()
        for i, data in enumerate(train_loader, 0):
            y, h_com, label = data['y'], data['h_com'], data['label']
            inputs = (y, h_com)

            # zero the parameter gradients
            detnet.zero_grad()

            # forward + backward + optimize
            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs1 = prenet1(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            x = x.unsqueeze(-1)
            x, h, outputs2 = detnet(inputs, x, h, STEP_SIZE, ITERATIONS)
            loss = tree_step2(outputs2, label, RATE)
            loss.backward()
            optim_det.step()
            detnet.r_cell.linear_h.rezeroWeights()
            detnet.r_cell.linear_x.rezeroWeights()

            # print statistics
            running_loss += loss.item()
            if i % (round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)) == round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE) - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                detnet.eval()
                prenet1.eval()
                y, h_com, label = data['y'], data['h_com'], data['label']
                inputs = (y, h_com)

                x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
                x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                x, h, outputs1 = prenet1(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
                x = x.unsqueeze(-1)
                x, h, outputs2 = detnet(inputs, x, h, STEP_SIZE, ITERATIONS)
                loss = tree_step2(outputs2, label, RATE)
                val_loss += loss.numpy()
                val_steps += 1
        print('validation loss: %.3f' % (val_loss / val_steps))
        scheduler.step()

    print('Training finished')
    # ----------------------------------------------- Test (STep 2)--------------------------------------------------
    with torch.no_grad():
        detnet.eval()
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
            x, h, outputs2 = detnet(inputs, x, h, STEP_SIZE, ITERATIONS)
            loss = tree_step2(outputs2, label, RATE)
            predictions += [x]
            test_loss += loss.numpy()
            test_steps += 1
        print('test loss: %.3f' % (test_loss / test_steps))
        predictions = torch.cat(predictions).cpu().numpy()

    # ------------------------------------- Weights Plot (Step 1)---------------------------------------------
    plot_part = predictions[:2000, :]
    style_size = 0.1
    points = plot_part
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.figure()
    plt.scatter(real_part, imag_part, style_size)

    points = plot_part[1]
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.scatter(real_part, imag_part, style_size)

    points = plot_part[2]
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.scatter(real_part, imag_part, style_size)

    points = plot_part[3]
    real_part = points[:, :TX]
    imag_part = points[:, TX:]
    plt.scatter(real_part, imag_part, style_size)

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.xticks([-3, -1, 1, 3])
    plt.yticks([-3, -1, 1, 3])

    plt.legend(['projection 2', 'projection 2', 'projection 3', 'projection 4'], loc=1)
    plt.title('Projection method for 16QAM/ Step2')
    plt.xlabel('Re', loc='right', fontdict={'fontsize': 'x-large', 'fontweight': 'bold'})
    plt.ylabel('Im', loc='bottom', fontdict={'fontsize': 'x-large', 'fontweight': 'bold', 'rotation': 'horizontal'})

    # ------------------------------------------------ Whole Test ------------------------------------------------------
    with torch.no_grad():
        detnet.eval()
        TEST_EBN0 = np.linspace(0, 15, 16)
        BER = []
        for ebn0 in TEST_EBN0:
            test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX,
                                                                                                  K=N_TEST, rate=RATE,
                                                                                                  EbN0=ebn0)
            test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)
            test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
            with torch.no_grad():
                detnet.eval()
                test_loss = 0.0
                test_steps = 0
                prediction = []
                for i, data in enumerate(test_loader, 0):
                    y, h_com, label = data['y'], data['h_com'], data['label']
                    inputs = (y, h_com)

                    x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
                    x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                    h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                    x, h, outputs1 = prenet1(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
                    x = x.unsqueeze(-1)
                    x, h, outputs2 = detnet(inputs, x, h, STEP_SIZE, ITERATIONS)
                    loss = tree_step2(outputs2, label, RATE)

                    prediction += [x]
                    test_loss += loss.numpy()
                    test_steps += 1
                print('test loss: %.3f' % (test_loss / test_steps))

                prediction = torch.cat(prediction).numpy()
                prediction = ((prediction + LENGTH - 1) / 2).round()
                ber = gray_ber(prediction, test_Data_real, test_Data_imag, rate=RATE)
                BER += [ber]
    # --------------------------------------- Save Model & Data --------------------------------------------------------
    PATH = '../../pretrained_projectionXTree/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f' % (TX, RX, RATE,
                                                                                                            EBN0_TRAIN,
                                                                                                            ITERATIONS,
                                                                                                            PROJECT_TIMES,
                                                                                                            BATCH_SIZE,
                                                                                                            RNN_HIDDEN_SIZE,
                                                                                                            STEP_SIZE)
    os.makedirs(PATH)
    data_ber = pd.DataFrame(BER, columns=['BER'])
    data_ber.to_csv(PATH+str('/ber.csv'))
    torch.save(detnet.state_dict(), PATH+str('/model_part2.pt'))
    # use the following line to load model
    detnet.load_state_dict(torch.load(PATH + str('/model.pt')))
