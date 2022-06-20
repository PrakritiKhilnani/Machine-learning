



import numpy as np

a=np.array([[1,2,3],[2,3,4]])
b=np.array([[2,3],[5,6],[2,1]])

# c=a*b
# X = np.c_[[1,1,1],[1,2,3]]
# print(X)
# print(a)
c=np.dot(a,b)
print(c)
