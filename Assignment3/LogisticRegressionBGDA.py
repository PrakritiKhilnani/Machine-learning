# This is a python code of Logistic Regression or Classification,it estimates an applicants probability of getting admission to an institution based on the scores
# of 2 examinations, without Feature Scaling.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_excel(r'D:\Prakriti\2 VSCode Python\ML\Book1.xlsx')
data=np.array(data)
data=data[1:,:]
df=pd.DataFrame(data)

x=df.iloc[:,0:2]
y=df.iloc[:,2]

# filter out the applicants that got admitted
admitted=df.loc[y==1]
# print(admitted)
 # filter out the applicants that din't get admission
not_admitted=df.loc[y==0]
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
# plt.show()

one=np.ones((len(x),1))
x=np.append(one,x,axis=1)
y=np.array(y).reshape((len(y),1))
#splitting the data, 70% for training and 30% for testing
split_pct=int(0.7*len(x))
#print(split_pct)
train_x, test_x = x[:split_pct], x[split_pct:]
train_y, test_y= y[:split_pct], y[split_pct:]

# defining functions to use
def sigmoid(z):
    return 1. / (1 + np.exp(-z))
def z(w, x):
    return np.dot(x,w.T) 
def hypothesis(w, x):
    return sigmoid(z(w, x))  
def costfunct(w,x,y):
    one_case=-y*np.log(hypothesis(w,x))
    zero_case=-(1-y)*np.log(1-hypothesis(w,x))
    cost= one_case+zero_case
    return (1/len(x))*sum(cost)
def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot((hypothesis(theta,x) - y).T,x)
def batch_gradient_descent(X,Y,learning_rate,iterations):
    cost_function = 0  # initalize our costfunct
    m=X.shape[0]
    theta=np.zeros(X.shape[1]).reshape(1,X.shape[1])
    for i in range(0,iterations):
       # prediction = Hypothesis  
        prediction =hypothesis(theta,X)
        # print(prediction)
        loss=prediction-Y

        theta=theta-learning_rate* sum(gradient(theta,X,Y))  
        # cost_function= costfunct(theta,X,Y)
        # print (f"Iteraton: {i},W={theta},mse={cost_function} ")     
    return theta
def accuracy(test_x,test_y,w,probab_threshold=0.5):
    predicted_classes = (hypothesis(w,test_x) >= probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten().astype(int)
    test_y=test_y.flatten().astype(int)
    # The following formula also gives the same result
    # h=predicted_classes==test_y
    # the sum() functn of an array with boolean values adds up only trues, i.e. 1
    # In this case 27 are correct predictions out of 30
    # accuracy = print((sum(h)/len(test_y))*100)
    accuracy = np.mean(predicted_classes == test_y)
    return accuracy * 100
# Finding parameters
# Learning rate=0.01 , epochs=50000
w=batch_gradient_descent(train_x,train_y,0.01,50000)
# Finding accuracy
acc=accuracy(test_x,test_y,w)
print(acc)
w=w.flatten()
print("The parameters are:", w)

# Taking two random values of X,
# to put in the equation of straight line to get the values of Y,
# Now having values of X and Y, plotting the graph.
x_values=np.linspace(30,100,2)
# to view X values
# print(x_values)
y_values = - (w[0] + np.dot(w[1], x_values)) / w[2]
# to view Y values
# print(y_values)
''' I could have use this too...to randomly pick X values for plotting the Decision Boundary
x_values = [np.min(x[:, 1] - 5), np.max(x[:, 2] + 5)]'''
plt.plot(x_values, y_values, label='Decision Boundary')
plt.show()

