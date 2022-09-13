import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

dir = "D:\\Dataset\\"
name = 'UMD'

test_file_path = dir + name + "\\" + name + "_TEST.csv"
train_file_path = dir + name + "\\" + name + "_TRAIN.csv"

df = pd.read_csv(test_file_path)
sil_score = []

for i in range(2, 9):
    kmeans_model = KMeans(n_clusters=i)
    kmeans_model.fit(df)
    sil_score.append(silhouette_score(df, kmeans_model.labels_))
X = range(2, 9)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('簇数')
plt.ylabel('轮廓系数值')
# plt.title(name+" "+"SC result")
plt.title(name+'轮廓系数结果')
plt.plot(X, sil_score, label="SC")
plt.legend()
plt.savefig('figsc1.svg', format='svg')
plt.show()
