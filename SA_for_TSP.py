import numpy as np
import pandas as pd
import random


'''
t表示初始温度，k表示温度下降速率，在（0，1）之间，l为某一温度水平下的迭代次数
'''


class sa_tsp():
    def __init__(self,data,t=1000,k=0.9,l=100,end_t=0.1):
        self.data = data
        self.t = t
        self.end_t = end_t
        self.k = k
        self.l = l
        self.n_dim = self.data.shape[0]

        # 初始化x
        self.x = list(range(0,self.n_dim))

        # 初始化x best
        self.x_best = self.x.copy()

    # 随机初始化
    def random_change(self):
        x_copy = self.x.copy()
        ran_num = random.sample(range(0,self.n_dim),4)
        self.x[ran_num[0]] = x_copy[ran_num[1]]
        self.x[ran_num[1]] = x_copy[ran_num[0]]
        self.x[ran_num[2]] = x_copy[ran_num[3]]
        self.x[ran_num[3]] = x_copy[ran_num[2]]

    # 计算距离
    def cal_func(self,x):
        dis = 0
        for i in range(len(x)-1):
            dis = self.data[x[i]][x[i+1]] + dis
        dis = dis + self.data[x[-1]][x[0]]
        return dis


    def run(self):
        while self.t > self.end_t:
            count = 0
            while count < self.l:
                count += 1
                self.random_change()
                d = self.cal_func(self.x) - self.cal_func(self.x_best)
                if d < 0:
                    self.x_best = self.x.copy()
                else:
                    if np.exp(-d / (self.k * self.t)) > np.random.uniform(0,1):
                        self.x_best = self.x.copy()
                    else:
                        self.x_best = self.x_best
            self.t = self.k * self.t
        print(self.x_best)
        print(self.cal_func(self.x_best))





if __name__ == '__main__':
    data = np.array([[0,14.14,36.05,15.81,41.23],
                    [14.14,0,22.36,7.07,30],
                    [36.05,22.36,0,25.49,14.14],
                    [15.81,7.07,25.49,0,35.35],
                    [41.23,30,14.14,35.35,0]])

    a = sa_tsp(data,t=1000,l=500,k=0.99)
    a.run()
