import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
from scipy import fftpack
from numpy.fft import *
from sklearn.preprocessing import StandardScaler


def data_load(train_path='data/train_features.csv', train_label_path='data/train_labels.csv',
              test_path='data/test_features.csv', sub_path='data/sample_submission.csv'):
    train = pd.read_csv(train_path)
    train_label = pd.read_csv(train_label_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)

    return train, train_label, test, sub


def make_model():
    model = keras.models.Sequential([
        keras.layers.Conv1D(128, 9, padding='same', input_shape=[600, 18]),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Conv1D(256, 6, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Conv1D(128, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(61, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def delete_output_layer(model):
    output = model.layers[-2].output
    model = keras.Model(inputs=[model.input], outputs=[output])

    return model


def add_feature(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train['acc_Energy'] = (train['acc_x'] ** 2 + train['acc_y'] ** 2 + train['acc_z'] ** 2) ** (1 / 3)
    test['acc_Energy'] = (test['acc_x'] ** 2 + test['acc_y'] ** 2 + test['acc_z'] ** 2) ** (1 / 3)

    train['gy_Energy'] = (train['gy_x'] ** 2 + train['gy_y'] ** 2 + train['gy_z'] ** 2) ** (1 / 3)
    test['gy_Energy'] = (test['gy_x'] ** 2 + test['gy_y'] ** 2 + test['gy_z'] ** 2) ** (1 / 3)

    train['gy_acc_Energy'] = ((train['gy_x'] - train['acc_x']) ** 2 + (train['gy_y'] - train['acc_y']) ** 2 + (
            train['gy_z'] - train['acc_z']) ** 2) ** (1 / 3)
    test['gy_acc_Energy'] = ((test['gy_x'] - test['acc_x']) ** 2 + (test['gy_y'] - test['acc_y']) ** 2 + (
            test['gy_z'] - test['acc_z']) ** 2) ** (1 / 3)

    train_dt = []
    for i in tqdm(train['id'].unique()):
        temp = train.loc[train['id'] == i]
        for v in train.columns[2:]:
            values = jerk_signal(temp[v].values)
            values = np.insert(values, 0, 0)
            temp.loc[:, v + '_dt'] = values
        train_dt.append(temp)

    test_dt = []
    for i in tqdm(test['id'].unique()):
        temp = test.loc[test['id'] == i]
        for v in train.columns[2:]:
            values = jerk_signal(temp[v].values)
            values = np.insert(values, 0, 0)
            temp.loc[:, v + '_dt'] = values
        test_dt.append(temp)

    train = pd.concat(train_dt)

    fft = []
    for i in tqdm(train['id'].unique()):
        temp = train.loc[train['id'] == i]
        for i in train.columns[2:8]:
            temp[i] = fourier_transform_one_signal(temp[i].values)
        fft.append(temp)
    train = pd.concat(fft)

    test = pd.concat(test_dt)

    fft_t = []
    for i in tqdm(test['id'].unique()):
        temp = test.loc[test['id'] == i]
        for i in test.columns[2:8]:
            temp[i] = fourier_transform_one_signal(temp[i].values)
        fft_t.append(temp)
    test = pd.concat(fft_t)

    col = train.columns
    train_s = train.copy()
    test_s = test.copy()

    scaler = StandardScaler()

    train_s.iloc[:, 2:] = scaler.fit_transform(train_s.iloc[:, 2:])
    train_sc = pd.DataFrame(data=train_s, columns=col)

    test_s.iloc[:, 2:] = scaler.transform(test_s.iloc[:, 2:])
    test_sc = pd.DataFrame(data=test_s, columns=col)

    return train, test


def fourier_transform_one_signal(t_signal):
    complex_f_signal = fftpack.fft(t_signal)
    amplitude_f_signal = np.abs(complex_f_signal)
    return amplitude_f_signal


def jerk_signal(signal, dt=0.9):
    return np.array([(signal[i + 1] - signal[i]) / dt for i in range(len(signal) - 1)])

    return train, test


def make_dataset(data):
    ids = data['id'].unique()
    id_data = data.groupby('id')
    series_data = []

    for i in ids:
        df = id_data.get_group(i)
        df = df.drop(['id', 'time'], axis=1)
        series_data.append(df.to_numpy())

    series_data = np.array(series_data)
    return series_data


def main(train_path, train_label_path, test_path, sub_path):
    train, label, test, sub = data_load(train_path, train_label_path, test_path, sub_path)
    model = make_model()
    model.load_weights('ckpt/best_setting.hdf5')    # cnn 모델의 weight load

    extractor = delete_output_layer(model)  # 마지막 Dense layer 를 제거

    train_sc, test_sc = add_feature(train, test)    # feature 추가

    train_series = make_dataset(train_sc)

    extract = extractor.predict(train_series)  # 600 * 18 벡터 --> 128 * 1 벡터
    extract = pd.DataFrame(extract)

    return extract
