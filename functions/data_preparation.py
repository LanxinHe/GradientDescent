import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder


def get_data(tx=8, rx=8, K=20000, rate=1, EbN0=15):
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # hty(K, 2tx), hth(K, 2tx, 2tx)
    hty = np.matmul(np.transpose(h_com, [2, 1, 0]), np.expand_dims(receive_signal_noised, axis=-1)).squeeze(-1)
    hth = np.matmul(np.transpose(h_com, [2, 1, 0]), np.transpose(h_com, [2, 0, 1]))

    return hty, hth, data_real, data_imag


def get_mmse(tx=8, rx=8, K=1000, rate=1, EbN0=15):
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])    # (2tx, K)

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # --------------------------------------------------- MMSE --------------------------------------------------------
    extended_part_real = np.sqrt(var*3/2/(l**2-1))*np.eye(tx)

    mmse_real = np.vstack([h_real, np.tile(np.expand_dims(extended_part_real, axis=-1), [1, 1, K])])    # (rx+tx, tx, K)
    mmse_imag = np.vstack([h_imag, np.zeros(([tx, tx, K]))])    # (rx+tx, tx, K)
    h_mmse = np.vstack([np.hstack([mmse_real, -1*mmse_imag]),
                        np.hstack([mmse_imag, mmse_real])])     # (2rx+2tx, 2tx, K)
    receive_zeros = np.zeros([K, tx])
    receive_mmse = np.hstack([receive_signal_noised[:, :rx], receive_zeros,
                              receive_signal_noised[:, rx:], receive_zeros])    # (K, 2rx+2tx)

    return receive_mmse, np.transpose(h_mmse, [2, 0, 1]), data_real, data_imag


def get_onehot_label(data_real, data_imag):
    cat = np.concatenate([data_real, data_imag])
    onehot = OneHotEncoder(sparse=False)
    label_oh = onehot.fit_transform(cat.T)
    return label_oh