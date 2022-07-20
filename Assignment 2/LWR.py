#     LOCALLY WEIGHTED REGRESSION(LWR) WITH TAU 0.8

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_excel (r'D:\Prakriti\2 VSCode Python\ML\Hp.xlsx') 
data=np.array(data)
Y=data[:,1].astype(int)
X=data[:,2:5].astype(int)

#because earlier the datatype was 'Object' as the excel sheet has string values too 
# X = X.astype('int32')
# Y= Y.astype('int32')

one = np.ones((len(X),1)) 
# can also use np.shape(X),instead of (len(x),1)
X = np.append(one, X, axis=1)

#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))


# function to calculate W weight diagonal Matrix used in calculation of predictions
def get_WeightMatrix(query_point,X, bandwidth):
  # M is the No of training examples
  M =X.shape[0]
  # Initialising W with identity matrix
  W = np.mat(np.eye(M))
  # calculating weights for query points
  for i in range(M):
    xi =X[i]
    denominator = (-2 * bandwidth**2)
    W[i, i] = np.exp(np.dot((xi-query_point), (xi-query_point).T)/denominator)
    return W
    
# function to return local weight of eah training example
def localWeight(query_point,X,Y, bandwidth):
    wt = get_WeightMatrix(query_point,X,bandwidth)
    W = (X.T * (wt*X)).I * (X.T * wt * Y)
    return W

# root function that drives the algorithm
def localWeightRegression(x, y, bandwidth):
    m,n = np.shape(x)
    ypred = np.zeros(m)
    
    for i in range(m):
        ypred[i] = x[i] * localWeight(x[i],x,y,bandwidth)
        
    return ypred

def mean_squared_error(y_true, y_predicted):
     # Calculating the loss or cost
    loss=y_predicted-y_true
    J= np.sum(loss**2) / (2*len(y_true))
    return J

# predicting values using LWLR
ypred = localWeightRegression(X, Y,0.8)
J=mean_squared_error(Y,ypred)
# print(f"Predicted y is:{ypred} and mse={J}")
print(ypred)
print(J)
plt.scatter(X[:,1],Y, color='blue')
plt.plot(X[:,1],ypred[X[:, 1].argsort(0)],color='yellow',linewidth=1)   
plt.show()
