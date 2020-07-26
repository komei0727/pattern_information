import pandas as pd
import numpy as np
from module import Linear_multiple_regression, normalization

#データの読み込み
auto_mpg_data =  pd.read_csv("auto-mpg.csv", encoding="utf-8").sample(frac = 1)
test_data = pd.read_csv("auto-mpg-test.csv", encoding="utf-8").sample(frac = 1)

#データを配列に変換
mpg = auto_mpg_data.loc[:, "mpg"].values
inputs_all = auto_mpg_data.loc[:, ["cylinders", "displacement", "weight", "acceleration", "model_year", "origin"]].values
mpg_test = test_data.loc[:, "mpg"].values
inputs_all_test = test_data.loc[:, ["cylinders", "displacement", "weight", "acceleration", "model_year", "origin"]].values

#正規化
mpg_n = normalization.normalization(mpg)
inputs_all_n = normalization.normalization(inputs_all)
inputs_all_test_n = normalization.normalization(inputs_all_test)

#線形重回帰の実行
w_all = Linear_multiple_regression.Linear_multiple_regression(inputs_all_n, mpg_n)

#決定係数の計算
s_all = np.sum(np.power(mpg_n - np.mean(mpg_n), 2))
s_res = np.sum(np.power(mpg_n - np.dot(inputs_all_n, w_all.T), 2))
R = 1 - (s_res / (len(mpg_n) - 6 - 1)) / (s_all / (len(mpg_n) - 1))

#予測値の計算
mpg_predict = np.dot(inputs_all_test_n, w_all.T) * np.std(mpg) + np.mean(mpg)

print("決定係数:{}".format(R))
print("mpg:", end="")
print(mpg_test)
print("mpg推定値:", end="")
print(mpg_predict)
