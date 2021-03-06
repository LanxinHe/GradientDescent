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
    PROJECT_TIMES = 4
    RNN_HIDDEN_SIZE = 6 * TX
    STEP_SIZE = 0.012
    ITERATIONS = 5

    train_y, train_h_com, train_Data_real, train_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_y, train_h_com, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # ------------------------------------- Establish Network ----------------------------------------------
    # PATH = '../../pretrained_projectionV/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f' % (TX, RX, RATE,
    #                                                                                                         EBN0_TRAIN,
    #                                                                                                         ITERATIONS,
    #                                                                                                         PROJECT_TIMES,
    #                                                                                                         BATCH_SIZE,
    #                                                                                                         RNN_HIDDEN_SIZE,
    #                                                                                                         STEP_SIZE)
    detnet = ProjectionX.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES)
    # detnet.load_state_dict(torch.load(PATH + str('/model.pt')))

    optim_det = torch.optim.Adam(detnet.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim_det, step_size=10, gamma=0.2)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_det, [10, 20, 35, 50, 70, 90], 0.1)

    # ------------------------------------- Train ----------------------------------------------------------
    print('Begin Training:')
    for epoch in range(EPOCHS):
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

            x, h, outputs = detnet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            loss = weighted_mse(outputs, label, RATE)
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
                # prenet1.eval()
                y, h_com, label = data['y'], data['h_com'], data['label']
                inputs = (y, h_com)

                x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
                x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                x, h, outputs = detnet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
                loss = weighted_mse(outputs, label, RATE)
                val_loss += loss.numpy()
                val_steps += 1
        print('validation loss: %.3f' % (val_loss / val_steps))
        scheduler.step()

    print('Training finished')

    # --------------------------------------------------- Test ---------------------------------------------------------
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

            x, h, outputs = detnet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            loss = weighted_mse(outputs, label, RATE)
            predictions += [x]
            test_loss += loss.numpy()
            test_steps += 1
        print('test loss: %.3f' % (test_loss / test_steps))
        predictions = torch.cat(predictions).cpu().numpy()
        predictions = ((predictions + LENGTH - 1)/2).round()
        ber = gray_ber(predictions, test_Data_real, test_Data_imag, rate=RATE)
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

                    x, h, outputs = detnet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)

                    loss = weighted_mse(outputs, label, RATE)
                    prediction += [x]
                    test_loss += loss.numpy()
                    test_steps += 1
                print('test loss: %.3f' % (test_loss / test_steps))

                prediction = torch.cat(prediction).numpy()
                prediction = ((prediction + LENGTH - 1) / 2).round()
                ber = gray_ber(prediction, test_Data_real, test_Data_imag, rate=RATE)
                BER += [ber]
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
