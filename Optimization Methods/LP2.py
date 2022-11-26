import numpy as np
from numpy.lib.twodim_base import diag
class Interior_Point_Method_For_QP(object):
    def __init__(self,A,b,C,d,epsilon):
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.epsilon = epsilon
    def solve(self):
        (self.m,self.n) = (self.C.shape[0],self.C.shape[1])
        # 初始化x,l,s
        x = np.ones(shape=(self.n, ))*10
        s = np.ones(shape=(self.m, )) # l为lambda
        v = np.ones(shape=(self.m, ))
        mu_k = np.dot(x, s)*0.1 / self.n
        # 辅助函数F_mu_t
        F = np.concatenate((np.dot(self.C,x)+s-self.d, np.dot(np.dot(diag(v),diag(s))-mu_k,np.ones(shape=self.m,)), np.dot(np.dot(self.A.T,self.A),x)-np.dot(self.A.T,self.b)+np.dot(self.C.T,v)))
        print(F)
        k = 0
        while np.linalg.norm(F, ord=2) > self.epsilon and k <= 1000:
            k += 1
            mu_k = np.dot(x, s)*0.1 / self.n
            (delta_x,delta_s,delta_v) = self.solve_delta(x,s,v,F,mu_k)
            alpha = self.linesearch(x,s,delta_x,delta_s,delta_v) #线搜索寻找步长
            (x,s,v) = (x + alpha * delta_x,s + alpha * delta_s,v + alpha * delta_v)
            print(x,s,v)
        return x
    def solve_delta(self,x,s,v,F,mu_k):
        A_ = np.zeros(shape=(self.m + self.m + self.n, self.m + self.m + self.n))
        A_[0:self.m, 0:self.m] = np.eye(self.m,self.m)
        A_[0:self.m, self.m+self.n:self.m + self.m + self.n] = np.copy(self.C)
        A_[self.m:self.m + self.m, self.m:self.m + self.m] = diag(s)
        A_[self.m:self.m + self.m, 0:self.m] = diag(v)
        A_[self.m + self.m:self.m + self.m + self.n, self.m:self.m + self.m] = np.copy(self.C.T)
        A_[self.m + self.m:self.m + self.m + self.n, self.m + self.m:self.m + self.m + self.n] = np.dot(self.A.T,self.A)

        r_ = -F
        # solve for delta
        delta = np.linalg.solve(A_, r_)
        delta_s = delta[0:self.m]
        delta_v = delta[self.m:self.m + self.m]
        delta_x = delta[self.m + self.m:self.m + self.m + self.n]
        return (delta_x,delta_s,delta_v)
    
    def linesearch(self,x,s,delta_x,delta_s,delta_v):
        alpha_max = 1.0
        for i in range(self.n):
            if delta_x[i] < 0:
                alpha_max = min(alpha_max, -x[i]/delta_x[i])
            if delta_s[i] < 0:
                alpha_max = min(alpha_max, -s[i]/delta_s[i])
        eta_k = 0.99
        return min(1.0, eta_k * alpha_max)

A = np.array([[2,-4],[0,4]])
b = np.array([-2,-6])
C = np.array([[1/2,1/2],[-1,2]])
d = np.array([1,2])
solver = Interior_Point_Method_For_QP(A,b,C,d,epsilon = 0.001)
print(solver.solve())