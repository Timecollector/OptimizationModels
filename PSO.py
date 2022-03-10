import numpy as np
import matplotlib.pyplot as plt

'''
粒子群算法
其中，n_dim为函数维度，pop_num为粒子数量，max_iter为迭代次数
lb为下界，ub为上界，如lb=[0,1],ub=[2,3]表示x1的定义域为[0,2],x2定义域为[1,3]
c1，c2，w为更新速度时的系数，取值范围为(0,1)
'''


class pso():
    def __init__(self, func, n_dim=2, pop_num=50, max_iter=100, lb=[0, 0], ub=[1, 1], c1=0.5, c2=0.5, w=0.9):
        self.func = func
        self.n_dim = n_dim
        self.pop_num = pop_num
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.ub = ub
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.best_y = []

        # 生成初始位置
        self.x_mat = np.zeros((self.n_dim, self.pop_num))
        for i in range(self.x_mat.shape[0]):
            self.x_mat[i] = np.random.uniform(self.lb[i], self.ub[i], (1, self.pop_num))

        # 初始化历史最优位置
        self.x_best_history = self.x_mat.copy()

        # 生成初始速度
        self.v_mat = np.zeros((self.n_dim, self.pop_num))
        for i in range(self.v_mat.shape[0]):
            self.v_mat[i] = np.random.uniform(self.lb[i], self.ub[i], (1, self.pop_num))

    # 更新当前粒子历史最优位置
    def update_p_best(self, i):
        if self.func(self.x_mat[:, i]) < self.func(self.x_best_history[:, i]):
            self.x_best_history[:, i] = self.x_mat[:, i]

    # 更新全局最优位置
    def update_g_best(self):
        g_best_mat = np.zeros((1, self.pop_num))[0]
        for i in range(self.pop_num):
            x = self.func(self.x_best_history[:, i])
            g_best_mat[i] = x
        g_best = np.where(g_best_mat == min(g_best_mat))
        return self.x_best_history[:, g_best[0][0]]

    # 定义速度更新函数
    def update_v(self, i):
        self.v_mat[:, i] = self.w * self.v_mat[:, i] + \
                           self.c1 * np.random.uniform(0, 1) * (self.x_best_history[:, i] - self.x_mat[:, i]) + \
                           self.c2 * np.random.uniform(0, 1) * (self.update_g_best() - self.x_mat[:, i])

    # 定义位置更新函数
    def update_x(self, i):
        self.x_mat[:, i] = self.x_mat[:, i] + self.v_mat[:, i]
        # 判断是否超出定义域
        for j in range(self.n_dim):
            if self.x_mat[j, i] < np.array(self.lb)[j]:
                self.x_mat[j, i] = np.array(self.lb)[j]
            if self.x_mat[j, i] > np.array(self.ub)[j]:
                self.x_mat[j, i] = np.array(self.ub)[j]

    # 定义运行函数
    def run(self):
        count = 0
        while count < self.max_iter:
            count += 1
            for i in range(self.pop_num):
                self.update_p_best(i)
                self.update_v(i)
                self.update_x(i)
            self.best_y.append(self.get_best_y())
        print('x_best: {0}'.format(self.update_g_best()))
        print('y_best: {0}'.format(self.func(self.update_g_best())))

    # 定义获取最优x的函数
    def get_best_x(self):
        return self.update_g_best()

    # 定义获取最优y的函数
    def get_best_y(self):
        return self.func(self.update_g_best())

    # 定义绘制迭代过程的函数
    def plot_best_y(self):
        plt.plot(range(self.max_iter), self.best_y)
        plt.xlabel('Epochs')
        plt.ylabel('Values')
        plt.grid(axis='y', ls='--')
        plt.show()

    # 想要自己绘制最优函数的话可以用这个获取迭代的y
    def get_y_history(self):
        return self.best_y


if __name__ == '__main__':
    def func(x):
        x1,x2,x3 = x
        return x1**2 - x1 * x2 + x2 * x3 - x1 * x3 + x1 * x2 * x3 + x2**2

    model = pso(func,n_dim=3,ub=[5,5,5],lb=[2,1,2])
    model.run()
