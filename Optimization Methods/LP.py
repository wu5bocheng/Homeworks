def newton(t, A, b, c, x0):
    maxIterationNumber = 50
    eps = 1.0e-8
    for count in range(maxIterationNumber):
        slack = 1.0/(A.dot(x0) - b)
        gradient = t*c - A.transpose().dot(slack) # 梯度
        H = A.transpose().dot(np.diag(np.square(slack))).dot(A) 
        delta = np.linalg.inv(H).dot(gradient)
        x0 = x0 - delta
        error = np.linalg.norm(delta)
        if (-error < eps):
            break
    return x0,error

def solver(t0, A, b, c, x0, factor):
    upperBound = 1000*t0
    t = [t0]
    solutions = [x0]
    eps = 1.0e-10
    while(t0 < upperBound):
        x0,error = newton(t0, A, b, c, x0)
        solutions.append(x0)
        print (x0)
        t0 = factor*t0
        t.append(t0)
        if (error < eps):
            break
    return x0