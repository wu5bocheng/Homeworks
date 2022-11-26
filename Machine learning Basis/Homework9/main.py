import numpy as np
from numpy.random import multivariate_normal


# alpha:隐变量权值（K）;mu多元高斯分布均值（n*K）;sigma多元高斯分布协方差矩阵（K*k）,len表示生成点数量
def make_data(len, alpha, mu, sigma):
    result_dic = {}
    result = np.zeros([len, 2])
    K = alpha.shape[0]
    for k in range(K):
        alpha_k = alpha[k]
        x_k = multivariate_normal(
            mean=mu[k], cov=sigma[k], size=(len), check_valid="raise")
        result_dic[alpha_k] = x_k
        result += alpha_k*x_k
    return result, result_dic


alpha = np.array([0.3, 0.3, 0.4])
mu = np.array([[3, 1], [8, 10], [12, 2]])
sigma = np.array(
    [[[1, -0.5], [-0.5, 1]], [[2, 0.8], [0.8, 2]], [[1, 0], [0, 1]]])
result, dic = make_data(300, alpha, mu, sigma)
