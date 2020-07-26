import numpy as np
from collections import Counter

def k_nearest(k, train_data, train_label, test_data):

    #テストデータと訓練データとの距離を計算し配列に格納
    distances = np.sum(np.power(train_data - test_data, 2), axis = 1)

    #距離の小さい順に並べ替え、インデックス番号を配列に格納
    indexes = np.argsort(distances)

    #インデックス番号の情報に基づき、距離の小さいものからk個を取り出す
    sorted_labels = train_label[indexes][:k]

    #最も多く出現したラベルを調べる
    label = Counter(sorted_labels).most_common(1)[0][0]

    return label

