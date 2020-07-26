import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from module import k_means, normalization

#データの読み込み
iris_data =  pd.read_csv("iris.csv", encoding="utf-8").sample(frac=1)

#データを配列に変換
datas = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values

#データの正規化
datas_n = normalization.normalization(datas)

k=5

#グラフ描画用のラベル
colors = ['blue','red','green','yellow','grey','black']
labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'centroid']

#グラフの描画設定
fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(8,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

centroids = np.zeros((6,k,2))
cluster, centroids[0] = k_means.k_means(k, datas_n[:,[0,1]], 1000)
while(not np.isnan(centroids).any() == False):
    cluster, centroids[0] = k_means.k_means(k, datas_n[:,[0,1]], 1000)

for i in range(k):
    axes[0,0].scatter(datas_n[cluster==i+1][:,0], datas_n[cluster==i+1][:,1], c=colors[i], label=labels[i])

axes[0,0].scatter(centroids[0][:,0],centroids[0][:,1],c=colors[-1],label=labels[-1],marker='^')
axes[0,0].set_title('SepalLength-SepalWidth')
axes[0,0].set_xlabel('SepalLength')
axes[0,0].set_ylabel('SepalWidth')
#axes[0,0].legend(loc='upper left')

cluster, centroids[1] = k_means.k_means(k, datas_n[:,[0,2]], 1000)
while(not np.isnan(centroids).any() == False):
    cluster, centroids[1] = k_means.k_means(k, datas_n[:,[0,2]], 1000)

for i in range(k):
    axes[0,1].scatter(datas_n[cluster==i+1][:,0], datas_n[cluster==i+1][:,2], c=colors[i], label=labels[i])

axes[0,1].scatter(centroids[1][:,0],centroids[1][:,1],c=colors[-1],label=labels[-1],marker='^')
axes[0,1].set_title('SepalLength-PetalLength')
axes[0,1].set_xlabel('SepalLength')
axes[0,1].set_ylabel('PetalLength')
#axes[0,1].legend(loc='upper left')

cluster, centroids[2] = k_means.k_means(k, datas_n[:,[0,3]], 1000)
while(not np.isnan(centroids).any() == False):
    cluster, centroids[2] = k_means.k_means(k, datas_n[:,[0,3]], 1000)

for i in range(k):
    axes[1,0].scatter(datas_n[cluster==i+1][:,0], datas_n[cluster==i+1][:,3], c=colors[i], label=labels[i])

axes[1,0].scatter(centroids[2][:,0],centroids[2][:,1],c=colors[-1],label=labels[-1],marker='^')
axes[1,0].set_title('SepalLength-PetalWidth')
axes[1,0].set_xlabel('SepalLength')
axes[1,0].set_ylabel('PetalWidth')
#axes[1,0].legend(loc='upper left')

cluster, centroids[3] = k_means.k_means(k, datas_n[:,[1,2]], 1000)
while(not np.isnan(centroids).any() == False):
    cluster, centroids[3] = k_means.k_means(k, datas_n[:,[1,2]], 1000)

for i in range(k):
    axes[1,1].scatter(datas_n[cluster==i+1][:,1], datas_n[cluster==i+1][:,2], c=colors[i], label=labels[i])

axes[1,1].scatter(centroids[3][:,0],centroids[3][:,1],c=colors[-1],label=labels[-1],marker='^')
axes[1,1].set_title('SepalWidth-PetalLength')
axes[1,1].set_xlabel('SepalWidth')
axes[1,1].set_ylabel('PetalLength')
#axes[1,1].legend(loc='upper left')

cluster, centroids[4] = k_means.k_means(k, datas_n[:,[1,3]], 1000)
while(not np.isnan(centroids).any() == False):
    cluster, centroids[4] = k_means.k_means(k, datas_n[:,[1,3]], 1000)

for i in range(k):
    axes[2,0].scatter(datas_n[cluster==i+1][:,1], datas_n[cluster==i+1][:,3], c=colors[i], label=labels[i])

axes[2,0].scatter(centroids[4][:,0],centroids[4][:,1],c=colors[-1],label=labels[-1],marker='^')
axes[2,0].set_title('SepalWidth-PetalWidth')
axes[2,0].set_xlabel('SepalWidth')
axes[2,0].set_ylabel('PetalWidth')
#axes[2,0].legend(loc='upper left')

cluster, centroids[5] = k_means.k_means(k, datas_n[:,[2,3]], 1000)
while(not np.isnan(centroids).any() == False):
    cluster, centroids[5] = k_means.k_means(k, datas_n[:,[2,3]], 1000)

for i in range(k):
    axes[2,1].scatter(datas_n[cluster==i+1][:,2], datas_n[cluster==i+1][:,3], c=colors[i], label=labels[i])

axes[2,1].scatter(centroids[5][:,0],centroids[5][:,1],c=colors[-1],label=labels[-1],marker='^')
axes[2,1].set_title('PetalLendth-PetalWidth')
axes[2,1].set_xlabel('PetalLength')
axes[2,1].set_ylabel('PetalWidth')
#axes[2,1].legend(loc='upper left')

plt.show()
