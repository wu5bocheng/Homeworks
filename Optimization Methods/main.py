import numpy as np
# 标准化后的矩阵输入初始化
class Simplex_Method(object):
    def __init__(self,A,b,c):
        self.A = A
        self.b = b
        self.c = -c #将min变为max
        self.tableau = np.array(np.concatenate((np.concatenate((A,b.T),axis=1),np.mat(np.append(self.c,0))),axis=0)) #单纯形表

    def solve(self):
        while self.can_be_improved():
            pivot_position = self.get_pivot_position()
            self.tableau = self.pivot_step(pivot_position)
        return self.get_solution()

    def can_be_improved(self): #基解可以被优化,终止条件
        self.z = self.tableau[-1,:-1]
        return any(x > 0 for x in self.z)

    def get_pivot_position(self): #找到更新位置
        column = next(i for i, x in enumerate(self.z) if x > 0)
        restrictions = []
        for eq in self.tableau[:-1,:]:
            el = eq[column]
            restrictions.append(np.inf if el <= 0 else eq[-1] / el)

        row = restrictions.index(min(restrictions))
        return row, column
    def pivot_step(self, pivot_position):# 更新单纯形表
        new_tableau = [[] for eq in self.tableau]
        i, j = pivot_position
        pivot_value = self.tableau[i][j]
        new_tableau[i] = np.array(self.tableau[i]) / pivot_value
        
        for eq_i, eq in enumerate(self.tableau):
            if eq_i != i:
                multiplier = np.array(new_tableau[i]) * self.tableau[eq_i][j]
                new_tableau[eq_i] = np.array(self.tableau[eq_i]) - multiplier
        return np.array(new_tableau)
    
    
    def is_basic(self,column):# 判断是否是基解
        return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1
    def get_solution(self):
        columns = np.array(self.tableau).T
        solutions = []
        for column in columns[:-1]:
            solution = 0
            if self.is_basic(column):
                one_index = column.tolist().index(1)
                solution = columns[-1][one_index]
            solutions.append(solution)
        return solutions


class Primal_Dual_Interior_Point_Method(object):
    def __init__(self,A,b,c,epsilon):
        self.A = A
        self.b = b
        self.c = c
        self.epsilon = epsilon
    def solve(self):
        (self.m,self.n) = (self.A.shape[0],self.A.shape[1])
        # 初始化x,l,s
        x = np.ones(shape=(self.n, ))
        l = np.ones(shape=(self.m, )) # l为lambda
        s = np.ones(shape=(self.n, ))
        k = 0
        while abs(np.dot(x, s)) > self.epsilon:
            k += 1
            sigma_k = 0.4 # 扰动KKT条件，sigma在（0,1）
            mu_k = np.dot(x, s) / self.n
            (delta_x,delta_l,delta_s) = self.solve_delta(x,l,s,sigma_k,mu_k)
            alpha = self.linesearch(x,s,delta_x,delta_l,delta_s) #线搜索寻找步长
            (x,l,s) = (x + alpha * delta_x,l + alpha * delta_l,s + alpha * delta_s)
        return x
    def solve_delta(self,x,l,s,sigma_k,mu_k):
        A_ = np.zeros(shape=(self.m + self.n + self.n, self.n + self.m + self.n))
        A_[0:self.m, 0:self.n] = np.copy(self.A)
        A_[self.m:self.m + self.n, self.n:self.n + self.m] = np.copy(self.A.T)
        A_[self.m:self.m + self.n, self.n + self.m:self.n + self.m + self.n] = np.eye(self.n)
        A_[self.m + self.n:self.m + self.n + self.n, 0:self.n] = np.copy(np.diag(s))
        A_[self.m + self.n:self.m + self.n + self.n, self.n + self.m:self.n + self.m + self.n] = np.copy(np.diag(x))

        r_ = np.zeros(shape=(self.n + self.m + self.n, ))
        r_[0:self.m] = np.copy(self.b - np.dot(self.A, x))
        r_[self.m:self.m + self.n] = np.copy(self.c - np.dot(self.A.T, l) - s)
        r_[self.m + self.n:self.m + self.n + self.n] = np.copy( sigma_k * mu_k * np.ones(shape=(self.n, )) - np.dot(np.dot(np.diag(x), np.diag(s)), np.ones(shape=(self.n, ))) )

        # solve for delta
        delta = np.linalg.solve(A_, r_)
        delta_x = delta[0:self.n]
        delta_l = delta[self.n:self.n + self.m]
        delta_s = delta[self.n + self.m:self.n + self.m + self.n]
        return (delta_x,delta_l,delta_s)
    
    def linesearch(self,x,s,delta_x,delta_l,delta_s):
        alpha_max = 1.0
        for i in range(self.n):
            if delta_x[i] < 0:
                alpha_max = min(alpha_max, -x[i]/delta_x[i])
            if delta_s[i] < 0:
                alpha_max = min(alpha_max, -s[i]/delta_s[i])
        eta_k = 0.99
        return min(1.0, eta_k * alpha_max)

A = np.array([[1,1,1,0],[2,1/2,0,1]])
b = np.array([[5,8]])
c = np.array([[-5,-1,0,0]])
simplex = Simplex_Method(A,b,c)
print("单纯形法的解：",simplex.solve())
Primal_Dual = Primal_Dual_Interior_Point_Method(A,b,c,0.0001)
print("原始-对偶内点法的解：",Primal_Dual.solve())

P = np.array([[2,-4],[0,4]])
q = np.array([-2,-6])
A = np.array([[1/2,1/2],[-1,2]])
b = np.array([1,2])