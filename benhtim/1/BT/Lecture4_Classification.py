# use pandas library to load data
import pandas as pd
dataset = pd.read_csv('D:\HK1_2023_2024\CT285\diabetes.csv')

# print the dataset size 
print('The size of dataset is: ', dataset.shape)

# get all variable in dataset 
all_var = dataset.columns.tolist()

# indicate X, Y 

ind_var = all_var[0:len(all_var)-1]
X = dataset[ind_var]
Y = dataset['Outcome'].values

#Split dataset into training and testing set (25%)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size =0.25)

# call algorithm: neural network
#from sklearn.neural_network import MLPClassifier
#model_nn = MLPClassifier(hidden_layer_sizes=(15,25,50))
#model_nn.fit(X_train,y_train)
#y_predict = model_nn.predict(X_test)

# call algorithm: Logistic regression
from sklearn.linear_model import LogisticRegression
model_lg = LogisticRegression()
model_lg.fit(X_train,y_train)
y_predict = model_lg.predict(X_test)


# Keep the result is stable 
import numpy as np
np.random.seed(0)

# Show the accurracy of prediction 
from sklearn.metrics import accuracy_score
print ('Accuracy of Neural network is: %2.f'%(100*accuracy_score(y_predict,y_test)),'%')

# Show the confusion matrix   
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
print('The confusion matrix is: \n',cm)

# Show the classification report 
from sklearn.metrics import classification_report
print ('The report is: \n',classification_report(y_predict,y_test))

