import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from module import k_means, normalization

#データの読み込み
iris_data =  pd.read_csv("iris.csv", encoding="utf-8").sample(frac=1)

#読み込んだデータを配列に変換
datas = iris_data.loc[:, ["PetalLength", "PetalWidth"]].values

#データの正規化
datas_n = normalization.normalization(datas)

#グラフ描画用の配列の準備
colors = ['blue','red','green','yellow','grey','black']
labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'centroid']

#グラフの描画設定
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

#グラフの位置を示す変数
p = 0
q = 0

#k=2~5でk-meansを行いグラフを描画
for k in range(2,6):

    #k_meansの実行
    cluster, centroids, = k_means.k_means(k, datas_n, 1000)
    #clusterが消滅した場合はやり直す
    while(not np.isnan(centroids).any() == False):
        cluster, centroids, = k_means.k_means(k, datas_n, 1000)

    #グラフの描画
    for i in range(k):
        axes[p,q].scatter(datas_n[cluster==i+1][:,0], datas_n[cluster==i+1][:,1], c=colors[i], label=labels[i])

    axes[p,q].scatter(centroids[:,0],centroids[:,1],c=colors[-1],label=labels[-1],marker='^')
    axes[p,q].set_title('k = %i' %k)
    axes[p,q].set_xlabel('PetalLength')
    axes[p,q].set_ylabel('PetalWidth')
    axes[p,q].legend(loc='upper left')

    #次のグラフの位置になるよう変数の値を更新
    if q == 0:
        q = 1
    else:
        q = 0
        p = 1

plt.show()


