
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn; import sklearn.datasets
# load image dataset: blue/red dots in circles
train_X, train_Y = sklearn.datasets.make_circles(n_samples=35, noise=.05)
A0=train_X.T; Y=train_Y.reshape(1,35)
# print(A0)
n0 = 2 ; n1 =3 ; n2 = 4; n3 = 1 ; m = 35
for i in range(35):
    if(Y[0][i]<0.5):
        plt.scatter(A0[0][i], A0[1][i],c="blue",marker="o")
    else:
        plt.scatter(A0[0][i], A0[1][i],c="purple",marker="x")
plt.title('x1, x2, y')
plt.show()
# A0 = abs(np.random.randn(n0,m)  )*0.01+0.7414
# Y = np.random.randint(2 , size = m)

WT1 = abs(np.random.randn(n1,n0) *0.01);
# b1 = abs(np.random.randn(n1,m))
b1 = abs(np.random.randn(n1,1))
WT2 = abs(np.random.randn(n2,n1) *0.01 )
# b2 = abs(np.random.randn(n2,m))
b2 = abs(np.random.randn(n2,1))
WT3 = abs(np.random.randn(n3,n2) *0.01);
# b3 = abs(np.random.randn(n3,m))
b3 = abs(np.random.randn(n3,1))
#
# plt.scatter(A0[0][0:13],A0[1][0:13],marker="o")
# plt.scatter(A0[0][13:35],A0[1][13:35],marker="x")
# plt.title("x1,x2,y")
# plt.show()
def g(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
def dg(Z):
    return 1-( g(Z) **2)

def forword_func(A0,WT1,WT2,WT3,b1,b2,b3):
    Z1 = np.dot(WT1,A0) + b1
    A1 = g(Z1)
    Z2 = np.dot(WT2,A1) + b2
    A2 = g(Z2)
    Z3 = np.dot(WT3,A2) + b3
    A3 = g(Z3)
    return Z1,Z2,Z3 , A1,A2,A3

def back_func(A0,A1,A2,A3,Y ,Z1,Z2,Z3):
    dA3 = -Y / A3 + (1-Y)* 1/(1-A3)
    dZ3 = dA3 * dg(Z3)
    dWT3 = 1/m*np.dot( dZ3 , A2.T )
    db3 = 1/m*np.sum(dZ3,axis=1,keepdims=True)
    # db3 = dZ3

    dA2 = np.dot(WT3.T,dZ3)
    dZ2 = dA2 * dg(Z2)
    dWT2 = 1/m*np.dot(dZ2, A1.T)
    # db2=dZ2
    db2 = 1/m*np.sum(dZ2,axis=1,keepdims=True)

    dA1 = np.dot(WT2.T,dZ2)
    dZ1 = dA1 * dg(Z1)
    dWT1 = 1/m*np.dot(dZ1, A0.T)
    # db1 = dZ1
    db1 = 1/m*np.sum(dZ1,axis=1,keepdims=True)

    return dWT1 ,dWT2,dWT3 ,db1,db2,db3
itera = 0
cost = [100]
alpha=0.8
while(abs(cost[0]) > 0.05 ):
    Z1,Z2,Z3,A1,A2,A3 = forword_func(A0,WT1,WT2,WT3,b1,b2,b3)
    cost = 1/m*np.sum((Y*np.log(abs(A3))+(1-Y)*np.log(abs(1-A3))) ,axis = 1)
    if(itera %10000 == 0):
        print("itera" , itera, "cost:" ,cost)
    if(itera==100000000000):
        break
    dWT1 ,dWT2,dWT3 ,db1,db2,db3 = back_func(A0,A1,A2,A3,Y ,Z1,Z2,Z3)
    WT3 = WT3 - alpha*dWT3 ; WT2 = WT2 - alpha*dWT2 ; WT1 = WT1 -alpha* dWT1
    b1 -=  alpha*db1 ;b2 -= alpha*db2 ; b3 -= alpha*db3
    itera += 1

    # print("A3:",A3)
    # print("Z3",Z3)

print("np.shape(Y):",np.shape(Y),"  np.shape(A3):",np.shape(A3))
