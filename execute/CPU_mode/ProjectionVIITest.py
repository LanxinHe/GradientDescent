from functions import data_preparation
from functions.test_functions import gray_ber
from functions.loss_cal import weighted_mse
from model.CPU_model import ProjectionVII
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
    RNN_HIDDEN_SIZE = 4 * TX
    STEP_SIZE = 0.012
    ITERATIONS = 6

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
    detnet = ProjectionVII.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES)
    # detnet.load_state_dict(torch.load(PATH + str('/model.pt')))

    optim_det = torch.optim.Adam(detnet.parameters(), lr=1e-3)
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
            # prenet1.zero_grad()

            # forward + backward + optimize
            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs = detnet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            loss = weighted_mse(outputs, label, RATE)
            loss.backward()
            optim_det.step()

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
    PATH = '../../pretrained_projectionVII/tx%i/rx%i/rate%i/EBN0_Train%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f' % (TX, RX, RATE,
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
    # detnet.load_state_dict(torch.load(PATH + str('/model1.pt')))

    # -------------------------------------- FineTune ------------------------------------------------------------------
    FINETUNE_RATE = 1e-9
    optim_pre1 = torch.optim.Adam(pre_det1.parameters(), lr=FINETUNE_RATE, weight_decay=0.001)
    scheduler_pre1 = torch.optim.lr_scheduler.ExponentialLR(optim_pre1, gamma=0.9)
    optim_pre2 = torch.optim.Adam(pre_det2.parameters(), lr=FINETUNE_RATE, weight_decay=0.001)
    scheduler_pre2 = torch.optim.lr_scheduler.ExponentialLR(optim_pre2, gamma=0.9)
    optim_pre3 = torch.optim.Adam(pre_det3.parameters(), lr=FINETUNE_RATE, weight_decay=0.001)
    scheduler_pre3 = torch.optim.lr_scheduler.ExponentialLR(optim_pre3, gamma=0.9)
    optim_pre4 = torch.optim.Adam(pre_det4.parameters(), lr=FINETUNE_RATE, weight_decay=0.001)
    scheduler_pre4 = torch.optim.lr_scheduler.ExponentialLR(optim_pre4, gamma=0.9)

    optim_det = torch.optim.Adam(detnet.parameters(), lr=FINETUNE_RATE, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_det, gamma=0.9)

    train_hty, train_hth, train_y, train_h_com, train_Data_real, train_Data_imag = data_with_channel.get_data(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    train_set = DetDataset(train_hty, train_hth, train_y, train_h_com)
    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)

    pre_det1.requires_grad_(True)
    pre_det1.train()
    pre_det2.requires_grad_(True)
    pre_det2.train()
    pre_det3.requires_grad_(True)
    pre_det3.train()
    pre_det4.requires_grad_(True)
    pre_det4.train()
    detnet.train()
    print('Begin Fine-tuning:')
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, y, h_com = (data['hty'], data['hth']), data['y'], data['h_com']

            # zero the parameter gradients
            pre_det1.zero_grad()
            pre_det2.zero_grad()
            pre_det3.zero_grad()
            pre_det4.zero_grad()
            detnet.zero_grad()

            # forward + backward + optimize
            prediction1, h1 = pre_det1(inputs, x_ini, h_ini)
            prediction2, h2 = pre_det2(inputs, prediction1[-1], h1)
            prediction3, h3 = pre_det3(inputs, prediction2[-1], h2)
            prediction4, h4 = pre_det4(inputs, prediction3[-1], h3)
            prediction5, h5 = detnet(inputs, prediction4[-1], h4)
            predictions = prediction1 + prediction2 + prediction3 + prediction4 + prediction5
            loss = ml_loss(predictions, y, h_com)
            loss.backward()
            optim_pre1.step()
            optim_pre2.step()
            optim_pre3.step()
            optim_pre4.step()
            optim_det.step()

            # print statistics
            running_loss += loss.item()
            if i % (round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)) == round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE) - 1:
                # writer.add_scalar('loss/train_loss', running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE), epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, y, h_com = (data['hty'], data['hth']), data['y'], data['h_com']
                prediction1, h1 = pre_det1(inputs, x_ini, h_ini)
                prediction2, h2 = pre_det2(inputs, prediction1[-1], h1)
                prediction3, h3 = pre_det3(inputs, prediction2[-1], h2)
                prediction4, h4 = pre_det4(inputs, prediction3[-1], h3)
                prediction5, h5 = detnet(inputs, prediction4[-1], h4)
                predictions = prediction1 + prediction2 + prediction3 + prediction4 + prediction5
                loss = ml_loss(predictions, y, h_com)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        print('validation loss: %.3f' % (val_loss / val_steps))
        scheduler_pre1.step()
        scheduler_pre2.step()
        scheduler_pre3.step()
        scheduler_pre4.step()
        scheduler.step()

    print('Fine-tuning finished')
