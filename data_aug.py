import pandas as pd
import numpy as np
from transforms3d.axangles import axangle2mat
import os


def make_series(data):
    ids = data['id'].unique()
    id_data = data.groupby('id')
    series_data = []

    for i in ids:
        df = id_data.get_group(i)
        df = df.drop(['id', 'time'], axis=1)
        series_data.append(df.to_numpy())

    series_data = np.array(series_data)
    return series_data


def exclude_ids(data):
    mask = data['label'] == 26
    return data.loc[mask, 'id'].tolist()


def rolling(data):
    choice_list = []
    sampling = np.random.choice(data.shape[0], int(data.shape[0] * 2 / 3))
    for j in sampling:
        while np.random.choice(data.shape[1]) not in choice_list:
            data[j] = np.roll(data[j], np.random.choice(data.shape[1]), axis=0)
            choice_list.append(np.random.choice(data.shape[1]))
    return data


def rotation(data):
    axis = np.random.uniform(low=-1, high=1, size=data.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(data, axangle2mat(axis, angle))


def permutation(data, nPerm=4, mSL=10):
    data_new = np.zeros(data.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(mSL, data.shape[0] - mSL, nPerm - 1))
        segs[-1] = data.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > mSL:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        data_temp = data[segs[idx[ii]]:segs[idx[ii] + 1], :]
        data_new[pp:pp + len(data_temp), :] = data_temp
        pp += len(data_temp)
    return data_new


# 우선은 rolling 만 해보자
def augmentation(x_train, y_train):
    series_train = make_series(x_train)
    rolling_data = rolling(series_train)
    return rolling_data


augmentation()