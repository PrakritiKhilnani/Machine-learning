#            NORMAL MINI BATCH GDA
import numpy as np
import pandas as pd 

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

#  splitting the data, 70% for training and 30% for testing
split_pct=int(0.7*len(X))
# print(split_pct)
train_x, test_x = X[:split_pct], X[split_pct:]
train_y, test_y= Y[:split_pct], Y[split_pct:]
print(train_y.dtype)

def mean_squared_error(y_true, y_predicted):
     # Calculating the loss or cost
    loss=y_predicted-y_true
    J= np.sum(loss**2) / (2*len(y_true))
    return J

def mini_batch_gradient_descent(x,y,iterations,learning_rate,batch_size=20):
    cost_function = 0  # initalize our functn
    theta = np.zeros(x.shape[1])
    shuffled_indices = np.random.permutation(len(y))
    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(iterations):
        xi = x_shuffled[i:i+batch_size]
        yi = y_shuffled[i:i+batch_size]

        prediction = np.dot(xi,theta.T)
        loss=prediction-yi
        cost_function= mean_squared_error(yi,prediction)
        gradient=sum(np.dot(loss.T,xi))
        # Updating the parameters i.e.theta here
        theta =theta - (learning_rate * gradient)/batch_size 
        print (f"Iteraton: {i},W={theta},mse={cost_function} ")
    return theta,cost_function   

W,J=mini_batch_gradient_descent(train_x,train_y,40,0.01)
# the value of parameters
print("The parameters are:", W) 
# the value of mean squared error
print("The value of cost function for the corresponding parameter is:", J)

