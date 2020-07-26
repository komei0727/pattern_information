import numpy as np

def normalization(datas):
    datas_n = (datas - np.mean(datas, axis = 0)) / np.std(datas, axis = 0)
    return datas_n
