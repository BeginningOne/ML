import numpy as np
import matplotlib.pyplot as plt

x = [4, 3, 3, 4, 2, 2, 0, 1, 2, 5, 1, 2, 5, 1, 3]
y = [8, 6, 6, 7, 4, 4, 2, 4, 5, 9, 3, 4, 8, 3, 6]


X = np.c_[np.ones(len(x)),x]
print(X)
y = np.c_[y]

def mov(theta):
    h = np.dot(X,theta)
    return h

def cos(h):
    j = 0.5*np.mean((h-y)**2)
    return j

def grad(sums=10000,alph=0.1):
    m,n = X.shape
    theta = np.zeros((n,1))
    j = np.zeros(sums)
    for i in range(sums):
        h = mov(theta)
        j[i] = cos(h)
        te = (1/m)*X.T.dot(h-y)
        theta -= alph * te
    return h,j,theta

# def score(h):
#     u = np.sum((h-y)**2)
#     v = np.sum((y-np.mean(y)))
#
#     return 1-u/v

if __name__ == '__main__':
    h,j,theta = grad()


    # plt.plot(j)
    # plt.show()
    #
    # plt.scatter(x,y,c='r')
    # plt.plot(x,h,c='b')
    # plt.show()
    #
    # print(theta)