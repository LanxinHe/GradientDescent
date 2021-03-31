import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def get_data(tx=8, rx=8, K=20000, rate=1, EbN0=15):
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(K, rx, tx)
    h_imag = np.sqrt(1/2) * np.random.randn(K, rx, tx)
    h_com = np.empty([K, 2 * rx, 2 * tx])
    for k in range(K):
        h_com[k, :, :] = np.vstack([np.hstack([h_real[k, :, :], -1 * h_imag[k, :, :]]),
                                    np.hstack([h_imag[k, :, :], h_real[k, :, :]])])

    noise_real = np.sqrt(1/2) * np.random.randn(rx, K)
    noise_imag = np.sqrt(1/2) * np.random.randn(rx, K)
    noise_com = np.vstack([noise_real, noise_imag])
    receive_signal = np.zeros([2*rx, K])
    for k in range(K):
        receive_signal[:, k] = np.dot(h_com[k, :, :], data_com[:, k])

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com

    h_t_y = np.empty([2 * tx, K])
    h_t_h = np.empty([K, 2 * tx, 2 * tx])
    for k in range(K):
        h_t_y[:, k] = np.dot(h_com[k, :, :].T, receive_signal_noised[:, k])
        h_t_h[k, :, :] = np.matmul(h_com[k, :, :].T, h_com[k, :, :])

    return h_t_y.T, h_t_h, data_real, data_imag


def get_mmse(tx=8, rx=8, K=1000, rate=1, EbN0=15):
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(K, rx, tx)
    h_imag = np.sqrt(1/2) * np.random.randn(K, rx, tx)
    h_com = np.empty([K, 2 * rx, 2 * tx])
    for k in range(K):
        h_com[k, :, :] = np.vstack([np.hstack([h_real[k, :, :], -1 * h_imag[k, :, :]]),
                                    np.hstack([h_imag[k, :, :], h_real[k, :, :]])])

    noise_real = np.sqrt(1/2) * np.random.randn(rx, K)
    noise_imag = np.sqrt(1/2) * np.random.randn(rx, K)
    noise_com = np.vstack([noise_real, noise_imag])
    receive_signal = np.zeros([2*rx, K])
    for k in range(K):
        receive_signal[:, k] = np.dot(h_com[k, :, :], data_com[:, k])

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com

    # --------------------------------------------------- MMSE --------------------------------------------------------
    extended_part_real = np.sqrt(var*3/2/(l^2-1))*np.eye(tx)
    extended_part_imag = np.zeros([rx, tx])
    mmse_real = np.empty([K, 2*rx, tx])
    mmse_imag = np.empty([K, 2*rx, tx])
    h_mmse = np.empty([K, 4*rx, 2*tx])
    receive_mmse = np.zeros([4*rx, K])
    for k in range(K):
        mmse_real[k] = np.vstack([h_real[k], extended_part_real])
        mmse_imag[k] = np.vstack([h_imag[k], extended_part_imag])
        h_mmse[k] = np.vstack([np.hstack([mmse_real[k], -1 * mmse_imag[k]]),
                               np.hstack([mmse_imag[k], mmse_real[k]])])
        receive_mmse[:rx, k] = receive_signal_noised[:rx, k]
        receive_mmse[2*rx:3*rx, k] = receive_signal_noised[rx:, k]

    h_t_y = np.empty([2 * tx, K])
    h_t_h = np.empty([K, 2 * tx, 2 * tx])
    for k in range(K):
        h_t_y[:, k] = np.dot(h_mmse[k, :, :].T, receive_mmse[:, k])
        h_t_h[k, :, :] = np.matmul(h_mmse[k, :, :].T, h_mmse[k, :, :])

    return h_t_y.T, h_t_h, receive_mmse.T, h_mmse, data_real, data_imag


def get_onehot_label(data_real, data_imag):
    cat = np.concatenate([data_real, data_imag])
    onehot = OneHotEncoder(sparse=False)
    label_oh = onehot.fit_transform(cat.T)
    return label_oh