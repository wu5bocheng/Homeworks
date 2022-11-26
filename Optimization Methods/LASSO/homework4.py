# %% [markdown]
# # 逻辑回归

# %% [markdown]
# ## SGD 固定步长
# 

# %%
import pandas as pd
import numpy as np
np.random.seed(2021)
#########数据预处理############
def make_data(dataset): #将数据处理成123维和
    m = dataset.shape[0]
    A = np.zeros([m,123])
    b = list(dataset[:,0].T)
    for i in range(m):
        for dics in dataset[i,1:]:
            if dics is not np.NaN:
                [n,value] = [int(dic) for dic in dics.split(":")] #将字符串分割，例如6:1表示下标为6的字符串的值为1
                A[i,n-1] = value
    return (A,b)
data = pd.read_table("a9a", header=None, delimiter=" ").iloc[:,:-1].values
(A,b) = make_data(data)

##########逻辑回归损失函数#########
def loss_function(A,b,x,lam):
    m = A.shape[1]
    return np.average([np.log(1 + np.exp(-b[i]*(A[i,:]@x))) for i in range(m)]) + lam*np.linalg.norm(x,ord=2)**2
def dfi(A,b,x,lam,i): #第i个分量梯度
    return 2*lam*x - np.mat(((np.exp(-b[i]*(A[i,:]@x))*b[i]*A[i,:])/(1 + np.exp(-b[i]*(A[i,:]@x))))).T

# %% [markdown]
# # 有固定步长的随机梯度下降法

# %%
def SGD_Fixed_Step(A,b,eps,step,lam):
    t = 0 #计数器
    s = step
    m = A.shape[0]
    x = np.zeros([A.shape[1], 1])# 初始值全0矩阵
    err = np.inf
    result_matrix = np.c_[t,x.T,loss_function(A,b,x,lam)]
    while (t < 1e4):
        origin_x = x
        i = np.random.randint(0,m-1)
        x = x - s*dfi(A,b,x,lam,i)
        fx = loss_function(A,b,x,lam)
        t += 1
        result_matrix = np.r_[result_matrix,np.c_[t,x.T,fx]] #结果存入矩阵方便画图
        # print (fx) #调试用代码
    f_star = min(result_matrix[:,-1])
    print("*"*100 + "\nlambda为：{lam}\n迭代次数为：{t}\n目标函数最优值为：{fx} \n最优解为：\n{x}\n".format(lam = lam,t = t, fx = f_star,x = list(x.T)))
    return result_matrix
colnames = ["iteration​"] + ["x_{}".format(i) for i in range(1,A.shape[1]+1)] + ["target_function"] #创建result_matrix的列名列表，形如：["iteration​","x_1","x_2",...,"x_100","target_function"]
result4 = SGD_Fixed_Step(A,b,eps = 1e-6,step = 1e-2,lam = 1e-2/A.shape[0]) #lam = 1e-2/N
pd.DataFrame(columns=colnames,data=result4).to_csv('result4.csv')

# %%
def SGD_Diminishing_Step(A,b,eps,step,lam):
    t = 0 #计数器
    s = step
    m = A.shape[0]
    x = np.zeros([A.shape[1], 1])# 初始值全0矩阵
    err = np.inf
    result_matrix = np.c_[t,x.T,loss_function(A,b,x,lam)]
    while (t < 1e4):
        origin_x = x
        i = np.random.randint(0,m-1)
        s = s*0.995
        x = x - s*dfi(A,b,x,lam,i)
        fx = loss_function(A,b,x,lam)
        t += 1
        result_matrix = np.r_[result_matrix,np.c_[t,x.T,fx]] #结果存入矩阵方便画图
        # print (fx) #调试用代码
    f_star = min(result_matrix[:,-1])
    print("*"*100 + "\nlambda为：{lam}\n迭代次数为：{t}\n目标函数最优值为：{fx} \n最优解为：\n{x}\n".format(lam = lam,t = t, fx = f_star,x = list(x.T)))
    return result_matrix
colnames = ["iteration​"] + ["x_{}".format(i) for i in range(1,A.shape[1]+1)] + ["target_function"] #创建result_matrix的列名列表，形如：["iteration​","x_1","x_2",...,"x_100","target_function"]
result5 = SGD_Diminishing_Step(A,b,eps = 1e-6,step = 1e-2,lam = 1e-2/A.shape[0]) #lam = 1e-2/N
pd.DataFrame(columns=colnames,data=result5).to_csv('result5.csv')

# %% [markdown]
# # SVRG

# %%
def SVRG(A,b,eps,learning_rate,lam,T):
    s = 0 #计数器
    step = learning_rate
    m = A.shape[0]
    x_tilde = np.zeros([A.shape[1], 1])# 初始值全0矩阵
    err = np.inf
    result_matrix = np.c_[s,x_tilde.T,loss_function(A,b,x_tilde,lam)]
    while (s < 1e4): #防止因为t选择0导致err = 0，直接弹出循环 
        origin_x = x_tilde
        z_tilde = np.average([dfi(A,b,x_tilde,lam,i) for i in range(m)])
        x = {0:x_tilde}
        for t in range(1,T+1): #进行T步迭代后计算一次全梯度     
            i = np.random.randint(0,m-1)
            x[t] = x[t-1] - step*(dfi(A,b,x[t-1],lam,i) - dfi(A,b,x_tilde,lam,i) + z_tilde)
        t = np.random.randint(0,T-1)
        x_tilde = x[t]
        fx = loss_function(A,b,x_tilde,lam)
        s += 1
        result_matrix = np.r_[result_matrix,np.c_[s,x_tilde.T,fx]] #结果存入矩阵方便画图
        # print (fx) #调试用代码
    f_star = min(result_matrix[:,-1])
    print("*"*100 + "\nlambda为：{lam}\n迭代次数为：{t}\n目标函数最优值为：{fx} \n最优解为：\n{x}\n".format(lam = lam,t = t, fx = f_star,x = list(x_tilde.T)))
    return result_matrix
colnames = ["iteration​"] + ["x_{}".format(i) for i in range(1,A.shape[1]+1)] + ["target_function"] #创建result_matrix的列名列表，形如：["iteration​","x_1","x_2",...,"x_100","target_function"]
result6 = SVRG(A,b,eps = 1e-6,learning_rate = 1e-2,lam = 1e-2/A.shape[0], T = 10) #lam = 1e-2/N
pd.DataFrame(columns=colnames,data=result6).to_csv('result6.csv')

# %% [markdown]
# # 画图

# %%
import matplotlib.pyplot as plt
def make_plot(result_matrix,label,color):
    x = [0.]+list(np.log(result_matrix[1:,0])) # 对x轴进行log采样
    plt.plot(x,result_matrix[:,-1],label = label,linewidth = 1, color = color)
plt.xlabel("Log Iteration Times")
plt.title("Converage of Target function and X\nlambda = {}".format("1e-2/N"))
make_plot(result4,"SGD with Fixed Learning Rate","black")
make_plot(result5,"SGD with Diminishing Learning Rate","blue")
make_plot(result6,"SVRG","red")
plt.legend()
plt.savefig("Converage of Logistic Regression.png",dpi=500)
plt.show()



