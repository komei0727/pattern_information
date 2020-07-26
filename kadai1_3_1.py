import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from module import Linear_multiple_regression, normalization
from mpl_toolkits.mplot3d import Axes3D

#データの取得
auto_mpg_data =  pd.read_csv("auto-mpg.csv", encoding="utf-8").sample(frac = 1)

#配列に変換
mpg = auto_mpg_data.loc[:, "mpg"].values
inputs = auto_mpg_data.loc[:, ["horsepower", "weight"]].values

#正規化
mpg_n = normalization.normalization(mpg)
inputs_n = normalization.normalization(inputs)

#線形重回帰の実行
w = Linear_multiple_regression.Linear_multiple_regression(inputs_n, mpg_n)

#グラフの描画
fig = plt.figure()

ax = Axes3D(fig)

ax.set_xlabel("horsepower")
ax.set_ylabel("weight")
ax.set_zlabel("mpg")

mesh_x1 = np.arange(inputs_n[:,0].min(), inputs_n[:,0].max(), (inputs_n[:,0].max()-inputs_n[:,0].min())/20)
mesh_x2 = np.arange(inputs_n[:,1].min(), inputs_n[:,1].max(), (inputs_n[:,1].max()-inputs_n[:,1].min())/20)
mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
mesh_y = w[0] * mesh_x1 + w[1] * mesh_x2 
ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y)
ax.plot(inputs_n[:,0],inputs_n[:,1],mpg_n,marker='o', c='black', Linestyle = 'None')

plt.show()

s_all = np.sum(np.power(mpg_n - np.mean(mpg_n), 2))
s_res = np.sum(np.power(mpg_n - np.dot(inputs_n, w.T), 2))
R = 1 - (s_res / (len(mpg_n) - 2 - 1)) / (s_all / (len(mpg_n) -1))

print("決定係数:{}".format(R))







