#                  BATCH GDA WITH FATURE SCALING
import numpy as np
import pandas as pd 

data = pd.read_excel (r'D:\Prakriti\2 VSCode Python\ML\Hp.xlsx') 
data=np.array(data)
Y=data[:,1]
X=data[:,2:5]

#because earlier the datatype was 'Object' as the excel sheet has string values too 
X = X.astype('float64')
Y= Y.astype('float64')
# Adding bias to feature matrix X
one = np.ones((len(X),1))
X = np.append(one, X, axis=1)
#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))

# FEATURE SCALING OF PLOT_SIZE
A=X[:,1]
A_norm= (A-min(A))/(max(A)-min(A))
# New feature matrix after Feature Scaling.
X_norm=np.c_[X[:,0],A_norm,X[:,2:4]] 
print(X_norm.shape)
#  splitting the data, 70% for training and 30% for testing
split_pct=int(0.7*len(X_norm))
# print(split_pct)

train_x, test_x = X_norm[:split_pct], X_norm[split_pct:]
train_y, test_y= Y[:split_pct], Y[split_pct:]

def mean_squared_error(y_true, y_predicted):
     # Calculating the loss or cost
    loss=y_predicted-y_true
    J= np.sum(loss**2) / (2*len(y_true))
    return J

def batch_gradient_descent(X,Y,learning_rate,iterations):
     cost_function = 0  # initalize our cost history list
     theta = np.zeros(X.shape[1])
     for i in range(0,iterations):
       # prediction = Hypothesis  
        prediction = np.dot(X,theta.T)
        loss=prediction-Y
        theta = theta - (learning_rate/len(Y)) * sum(np.dot(loss.T,X) )  
        cost_function= mean_squared_error(Y,prediction)
        print (f"Iteraton: {i},W={theta},mse={mean_squared_error(Y,prediction)} ")              
     return theta,cost_function

W,J=batch_gradient_descent(train_x,train_y,0.0001,25)
# the value of parameters
print("The parameters are:", W) 
# the value of mean squared error
print("The value of cost function for the corresponding parameter is:", J)