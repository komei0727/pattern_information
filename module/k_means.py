import numpy as np

def k_means(k, datas, times):
    #データの各要素がどのクラスターに属するか記録する配列を用意し、その各要素を1からkまでの整数で初期化
    cluster_list = list(range(1,k+1))
    cluster = np.random.choice(cluster_list, len(datas))

    #クラスターごとの重心を記録する配列の作成
    centroids = np.zeros((k,len(datas[0])))
    centroids_next = np.zeros((k,len(datas[0])))

    #クラスターごとの重心の計算
    for i in range(k):
        centroids[i] = np.mean(datas[cluster == i + 1], axis = 0)

    #クラスターの更新と重心の計算を繰り返す
    for t in range(times):
        for j in range(len(datas)):
            #各クラスターの重心との距離を計算し、一番距離の近かったクラスターに追加
            distances = np.sum(np.power(centroids - datas[j], 2),axis = 1)
            cluster[j] = np.argsort(distances)[0] + 1

        #重心の再計算
        for i in range(k):
            centroids_next[i] = np.mean(datas[cluster == i + 1], axis = 0)

        #重心の位置が変わらなくなれば終了
        if np.sum(centroids == centroids_next) == k:
            break
        centroids = centroids_next

    return cluster, centroids


