import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import k_nearest, normalization

#iris_dataを取得しランダムに並べ替える
iris_data = pd.read_csv("iris.csv", encoding="utf-8").sample(frac=1)

#データとラベルにわけ配列の形に変換する
datas = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values
labels = iris_data.loc[:, "Name"].values

#データを正規化
datas_n = normalization.normalization(datas)

#kの値を1から30まで変えてk近傍法を行う
accuracies = []
for k in range(30):
    count = 0
    #一つ抜き法
    for i in range(len(datas)):
        test_data = datas_n[i]
        train_data = np.delete(datas_n, i, 0)
        test_label = labels[i]
        train_label = np.delete(labels, i)
        label = k_nearest.k_nearest(k+1, train_data, train_label, test_data)

        #k近傍法による推定値と実際のラベルが等しければカウントを追加
        if label == test_label:
            count = count + 1

    #精度の計算
    accuracy = count / len(datas)
    accuracies.append(accuracy)

#グラフを描画する
k = list(range(1,31)) 
plt.plot(k, accuracies)
plt.title("k_nearest")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()


