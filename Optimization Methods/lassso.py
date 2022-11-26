#Proximal Gradient Descent算法求解LASSO问题

#导入要用的库
import numpy as np

#定义LASSO问题中的目标函数
def LASSOfunc(A, x, b, lambda1):
    """
    :param A:输入一个ndarray数组，表示LASSO问题中的系数矩阵A
    :param x:输入一个ndarray数组，表示LASSO问题的解
    :param b:输入一个ndarray数组，表示LASSO问题的常数向量b
    :param lambda1:输入一个数字，表示LASSO问题中正则化系数的值
    :return:返回一个数字，表示LASSO问题目标函数的值
    """
    y = (1/2)*np.linalg.norm(np.matmul(A, x)-b, ord=2)**2+lambda1*np.linalg.norm(x, ord=1)
    return y

#定义更新z^{t+1}的函数
def get_update_z(A, x, b, lambda1):
    """
    :param A:输入一个ndarray数组，表示LASSO问题中的系数矩阵A
    :param x:输入一个ndarray数组，表示LASSO问题的解，也是Proximal Gradient Descent算法中的x
    :param b:输入一个ndarray数组，表示LASSO问题的常数向量b
    :param lambda1:输入一个数字，表示LASSO问题中正则化系数的值
    :return: 返回一个ndarray数组，表示Proximal Gradient Descent算法中的z
    """
    matrix_I = np.eye(x.shape[0])
    ATA = np.matmul(A.T, A)
    ATA_maxeigv = max(np.linalg.eigvals(ATA))
    z = np.matmul((matrix_I - 1 / ATA_maxeigv * ATA), x) + 1 / ATA_maxeigv * np.matmul(A.T, b)
    return z

#定义更新x^{t+1}的函数
def get_update_x(A, z, b, lambda1):
    """
    :param A:输入一个ndarray数组，表示LASSO问题中的系数矩阵A
    :param z:输入一个ndarray数组，表示Proximal Gradient Descent算法中的z
    :param b:输入一个ndarray数组，表示LASSO问题的常数向量b
    :param lambda1:输入一个数字，表示LASSO问题中正则化系数的值
    :return: 返回一个ndarray数组，表示更新后LASSO问题的解，也是Proximal Gradient Descent算法中的x
    """
    ATA = np.matmul(A.T, A)
    ATA_maxeigv = max(np.linalg.eigvals(ATA))
    x = np.array([0.0 for i in range(z.shape[0])]).T
    x = np.reshape(x, (z.shape[0], 1))
    for i in range(z.shape[0]):
        if ATA_maxeigv * z[i][0] > lambda1:
            x[i][0] = z[i][0] - lambda1 / ATA_maxeigv
        elif ATA_maxeigv * z[i][0] < -lambda1:
            x[i][0] = z[i][0] + lambda1 / ATA_maxeigv
        else:
            x[i][0] = 0
    return x

#设置LASSO问题的参数
np.random.seed(2021) # set a constant seed to get same random matrixs
A = np.random.rand(500, 100)
x_ = np.zeros([100, 1])
x_[:5, 0] += np.array([i+1 for i in range(5)]) # x_ denotes expected x
b = np.matmul(A, x_) + np.random.randn(500, 1) * 0.1 # add a noise to b
lam = 0.1 # try some different values in {0.1, 1, 10}

#lambda=0.1时
#设置初始解
x = np.array([0.1 for i in range(100)]).T
x = np.reshape(x, (100, 1))
z = np.array([0.1 for i in range(100)]).T
z = np.reshape(z, (100, 1))
lam = 0.1

#开始迭代
t = 0
for i in range(2000):
    z = get_update_z(A, x, b, lam)
    x = get_update_x(A, z, b, lam)
    y = LASSOfunc(A, x, b, lam)
    t += 1
    #print("这是第{}轮迭代,现在目标函数的值是{}".format(t, y))
    #print("现在的解是", x.T)
print("共迭代了{}次".format(t))
print("最终的近似最优解为：", x.T)
print("最终的近似最优值为：", y)

#lambda=1时
#设置初始解
x = np.array([0.1 for i in range(100)]).T
x = np.reshape(x, (100, 1))
z = np.array([0.1 for i in range(100)]).T
z = np.reshape(z, (100, 1))
lam = 1

#开始迭代
t = 0
for i in range(2000):
    z = get_update_z(A, x, b, lam)
    x = get_update_x(A, z, b, lam)
    y = LASSOfunc(A, x, b, lam)
    t += 1
    #print("这是第{}轮迭代,现在目标函数的值是{}".format(t, y))
    #print("现在的解是", x.T)
print("共迭代了{}次".format(t))
print("最终的近似最优解为：", x.T)
print("最终的近似最优值为：", y)

#lambda=10时
#设置初始解
x = np.array([0.1 for i in range(100)]).T
x = np.reshape(x, (100, 1))
z = np.array([0.1 for i in range(100)]).T
z = np.reshape(z, (100, 1))
lam = 10

#开始迭代
t = 0
for i in range(2000):
    z = get_update_z(A, x, b, lam)
    x = get_update_x(A, z, b, lam)
    y = LASSOfunc(A, x, b, lam)
    t += 1
    print("这是第{}轮迭代,现在目标函数的值是{}".format(t, y))
    print("现在的解是", x.T)
print("共迭代了{}次".format(t))
print("最终的近似最优解为：", x.T)
print("最终的近似最优值为：", y)
