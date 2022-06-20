import numpy as np
import pandas as pd

data = pd.read_excel (r'D:\Prakriti\2 VSCode Python\ML\Hp.xlsx') 
data=np.array(data)

# np.random.shuffle(data)
# print(data)
Y=data[:,1]
X=data[:,2:5]

#because earlier the datatype was 'Object' as the excel sheet has string values too 
X = X.astype('float64')
Y= Y.astype('float64')
'''  But for the matrix multiplication we have to make it of the order
 (546 X 4) by adding a column of ones first(the bias( x0) term). Also,
  our Y should be a column vector (546 X 1)which is now in the form of a row vector.'''
 #adding ones to X
one = np.ones((len(X),1))
X = np.append(one, X, axis=1)

#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))
 
#  splitting the data, 70% for training and 30% for testing
split_pct=int(0.7*len(X))
print(split_pct)

train_x, test_x = X[:split_pct], X[split_pct:]
train_y, test_y= Y[:split_pct], Y[split_pct:]

#Function1
def normal_equation(X, Y):
     beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))
     return beta
# Function2
def predict(X, beta):
    return np.dot(X, beta)

# Using 70% of the data for training.
beta = normal_equation(train_x, train_y)
#predicting the value of y for all test samples,i.e. test data X,(Y=X*beta)
predictions = predict(test_x, beta)
# For checking the dimensions of the matrix/ to know I'm right or wrong
print(predictions.shape)  

# Function3
def metrics(predictions,test_y):

    #calculating mean absolute error
    MAE = np.mean(np.abs(predictions-test_y))

mae = metrics(predictions, test_y)
print("Mean Absolute Error: ", mae)

