import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score
from sklearn.metrics import normalized_mutual_info_score

class ts_cluster(object):
    def __init__(self, num_clust):
        self.num_clust = num_clust
        self.assignments = {}
        self.centroids = []

    def nearest(self, point, cluster_centers):
        min_dist = float('inf')
        m = np.shape(cluster_centers)[0]
        for i in range(m):
            d = self.eudistance(point, cluster_centers[i, ])
            if min_dist > d:
                min_dist = d
        return min_dist

    def get_cen(self, dataset, k):
        m, n = np.shape(dataset)
        self.centroids = np.zeros((k, n))
        index = np.random.randint(0, m)
        self.centroids[0,] = dataset[index,]
        d = [0.0 for _ in range(m)]
        for i in range(1, k):
            sum_all = 0
            for j in range(m):
                d[j] = self.nearest(dataset[j,], self.centroids[0:i, ])
                d[j] = d[j] ** 2
                sum_all += d[j]
            sum_all *= random.uniform(1/k, 1)
            for j, di in enumerate(d):
                sum_all = sum_all - di
                if sum_all > 0:
                    continue
                self.centroids[i,] = dataset[j,]
                break
        return self.centroids


    def k_means_clust(self, data, num_iter, w, progress=True):
        # if isinstance(data, np.ndarray):
        #     ind = random.sample(range(len(data)), k=self.num_clust)
        #     # ind = [18,32,54,83]
        #     self.centroids = data[ind]
        #     # ind1 = random.choice(range(10))
        #     # ind2 = random.choice(range(10,20))
        #     # ind3 = random.choice(range(20,30))
        #     # self.centroids = data[[0,10,21]]
        # else:
        #     self.centroids = random.sample(data, self.num_clust)

        self.centroids = self.get_cen(data, self.num_clust)
        # for i in self.centroids:
        #     # plt.ylim(-2.3, 3)
        #     plt.plot(i)
        # plt.show()

        for n in range(num_iter):
            if progress:
                print('iteration ' + str(n + 1))
            # assign data points to clusters
            self.assignments = {}
            for init in range(self.num_clust):
                self.assignments[init] = []
            for ind, i in enumerate(data):
                min_dist = float('inf')
                closest_clust = None
                for c_ind, j in enumerate(self.centroids):
                    # if self.LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = self.eudistance(i, j)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
                #     cur_dist = self.eudistance(i, j)
                #     if cur_dist < min_dist:
                #         min_dist = cur_dist
                #         closest_clust = c_ind
                # self.assignments[closest_clust].append(ind)
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust] = []

        for key in self.assignments:
            clust_sum = []
            for k in self.assignments[key]:
                d_s = data[k]
                clust_sum.append(d_s)
            if len(clust_sum) == 0:
                clust_sum = np.zeros((1, np.shape(data)[1]))
            clust_sum = pd.DataFrame(clust_sum)
            # plt.show()
            try:
                self.centroids[key] = clust_sum.mean().to_numpy().tolist()
            except:
                print(clust_sum)

    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def plot_centroids(self):
        for i in self.centroids:
            # plt.ylim(-2.3, 3)
            plt.plot(i)
        plt.show()

    def eudistance(self, s1, s2):
        d = 0
        for i in range(len(s1)):
            d += (s1[i] - s2[i]) ** 2
        return np.sqrt(d)

    def predict(self, data):
        ans = []
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            # wi = np.mean(self.weight, axis=0)
            for c_ind, j in enumerate(self.centroids):
                dtw_dist = self.eudistance(j, i)
                if dtw_dist < min_dist:
                    closest_clust = c_ind
                    min_dist = dtw_dist
            ans.append(closest_clust+1)
        return ans


dir = "D:\\Dataset\\"
name = 'Plane'

RI_test = []
RI_train = []
NMI_test = []
NMI_train = []
Dis = []
random.seed(12)

for iteration in range(10):
    print("training " + name + ":")
    test_file_path = dir + name + "\\" + name + "_TEST.csv"
    train_file_path = dir + name + "\\" + name + "_TRAIN.csv"
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)


    leng = len(train_data.columns)
    train_data.columns = range(1, leng + 1)
    test_data.columns = range(1, leng + 1)
    train_data.sort_values(by=1, inplace=True)
    train_data = train_data.reset_index(drop=True)

    test_data.sort_values(by=1, inplace=True)
    test_data = test_data.reset_index(drop=True)

    cluster_nums = len(list(set(train_data.iloc[:, 0])))

    label_train = train_data[1].values
    data_train = train_data.iloc[:, 1:].values

    label_test = test_data[1].values
    data_test = test_data.iloc[:, 1:].values

    # print(label_train)

    iter_times = 1
    k_means_clust = ts_cluster(cluster_nums)
    k_means_clust.k_means_clust(data=data_train, num_iter=iter_times, w=None)

    label_p_train = k_means_clust.predict(data_train)
    label_p_test = k_means_clust.predict(data_test)

    print("calculating RI and NMI:")
    RI = rand_score(label_train, label_p_train)
    RI_train.append(RI)
    RI = rand_score(label_test, label_p_test)
    RI_test.append(RI)

    NMI = normalized_mutual_info_score(label_train, label_p_train)
    NMI_train.append(NMI)
    NMI = normalized_mutual_info_score(label_test, label_p_test)
    NMI_test.append(NMI)

    # k_means_clust.plot_centroids()

    c = k_means_clust.get_centroids()
    d = 0
    for i in range(0, cluster_nums):
        for j in range(i + 1, cluster_nums):
            d += k_means_clust.eudistance(c[i], c[j])
    Dis.append(d)
    # print(c)
    # print("中心距离和：{}".format(d))

print(RI_train, RI_test)
print(NMI_train, NMI_test)
AVE_RI1 = np.mean(RI_train)
AVE_RI2 = np.mean(RI_test)
STD_RI1 = np.std(RI_train)
STD_RI2 = np.std(RI_test)
AVE_NMI1 = np.mean(NMI_train)
AVE_NMI2 = np.mean(NMI_test)
STD_NMI1 = np.std(NMI_train)
STD_NMI2 = np.std(NMI_test)
print("直接：{} +/- {}".format(AVE_RI1, STD_RI1))
print("按结果：{} +/- {}".format(AVE_RI2, STD_RI2))
print("直接：{} +/- {}".format(AVE_NMI1, STD_NMI1))
print("按结果：{} +/- {}".format(AVE_NMI2, STD_NMI2))

DIS = np.mean(Dis)
print("中心距离和：{}".format(DIS))
