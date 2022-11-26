for t in range(1,10): #进行T步迭代后计算一次全梯度     
    x = {0:1+t}
    print(t)
    x[t] = x[t-1] - t
    print(x)