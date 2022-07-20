#                                   NORMAL BATCH GDA
import numpy as np
import pandas as pd 

data = pd.read_excel (r'D:\Prakriti\2 VSCode Python\ML\Hp.xlsx') 
data=np.array(data)
Y=data[:,1]
X=data[:,2:5]

#because earlier the datatype was 'Object' as the excel sheet has string values too 
X = X.astype('float64')
Y= Y.astype('float64')

one = np.ones((len(X),1)) 
# can also use np.shape(X),instead of (len(x),1)
X = np.append(one, X, axis=1)

#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))

#  splitting the data, 70% for training and 30% for testing
split_pct=int(0.7*len(X))
# print(split_pct)

train_x, test_x = X[:split_pct], X[split_pct:]
train_y, test_y= Y[:split_pct], Y[split_pct:]

def mean_squared_error(y_true, y_predicted):
     # Calculating the loss or cost
    loss=y_predicted-y_true
    J= np.sum(loss**2) / (2*len(y_true))
    return J
# Gradient Descent Function
# Here iterations, learning_rate
# are hyperparameters that can be tuned
def batch_gradient_descent(X,Y,learning_rate,iterations):
     cost_function = 0  # initalize our cost history list
     theta = np.zeros(X.shape[1])
     for i in range(0,iterations):
       # prediction = Hypothesis  
        prediction = np.dot(X,theta.T)
        loss=prediction-Y
        cost_function= mean_squared_error(Y,prediction)
        # Updating the parameters i.e.theta here
        theta = theta - (learning_rate/len(Y)) * sum(np.dot(loss.T,X))  
        print (f"Iteraton: {i},W={theta},mse={cost_function} ")              
     return theta,cost_function

W,J=batch_gradient_descent(train_x,train_y,0.0001,25)
# the value of parameters
print("The parameters are:", W) 
# the value of mean squared error
print("The value of cost function for the corresponding parameter is:", J)
