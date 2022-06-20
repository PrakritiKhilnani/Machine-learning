import math
import csv
import random
import numpy as np
import pandas as pd

data = pd.read_excel (r'D:\Prakriti\Hp.xlsx') 
# df = pd.DataFrame(data, columns= ['price'])
# print (df)
print(data)

data=np.array(data)

Y=data[:,1]
# print(X1)
X=data[:,2:5]

'''  But for the matrix multiplication we have to make it of the order
 (546 X 4) by adding a column of ones first(the bias( x0) term). Also,
  our Y should be a column vector (546 X 1)which is now in the form of a row vector.'''
 #adding ones to X
one = np.ones((len(X),1))
X = np.append(one, X, axis=1)

#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))
# print(X.shape)
# print(Y.shape)
print(X)
# print(Y)

# np.random.shuffle(X)
# print(X)
