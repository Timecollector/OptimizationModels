import numpy as np

'''
t表示初始温度，k表示温度下降速率，在（0，1）之间，l为某一温度水平下的迭代次数
'''


class sa():
    def __init__(self, func, n_dim=2, t=100, k=0.9, lb=[0, 0], ub=[1, 1], l=10, end_t=10, acc=0.01):
        self.func = func
        self.t = t
        self.k = k
        self.lb = lb
        self.ub = ub
        self.n_dim = n_dim
        self.l = l
        self.end_t = end_t
        self.acc = acc

        # 随机产生初始解
        self.x = np.zeros((1, self.n_dim))
        for i in range(self.n_dim):
            self.x[0][i] = np.random.uniform(lb[i], ub[i], (1, 1))

        # 定义最优解
        self.x_best = self.x.copy()

    # 产生新解
    def random_per(self):
        per = np.zeros((1, self.n_dim))
        for i in range(self.n_dim):
            per[0][i] = np.random.uniform(-1, 1, (1, 1))
        self.x = self.x + self.k * per
        for i in range(self.n_dim):
            if self.x[0][i] < self.lb[i]:
                self.x[0][i] = self.lb[i]
            elif self.x[0][i] > self.ub[i]:
                self.x[0][i] = self.ub[i]

    def run(self):
        while self.t > self.end_t:
            count = 0
            while count < self.l:
                count += 1
                self.random_per()
                d = self.func(self.x[0]) - self.func(self.x_best[0])
                if d <= 0:
                    self.x_best = self.x.copy()
                else:
                    if np.exp(-d / (self.k * self.t)) > np.random.uniform(0, 1):
                        self.x_best = self.x.copy()
                    else:
                        self.x_best = self.x_best
                self.x = self.x_best.copy()
            self.t = self.k * self.t
        print(self.x_best)
        print(self.func(self.x_best[0]))
