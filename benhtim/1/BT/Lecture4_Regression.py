# use pandas library to load data. Please place dataset fish_market.csv at D:/
# pandasライブラリを使用してデータをロードします。 データセットfish_market.csvをD：/に配置してください
import pandas as pd
# dataset = pd.read_csv('D:\\HK2_2022_2023\BI\\insurance.csv')
dataset =pd.read_csv('insurance.csv')
# =============================================================================
# Find X and Y in dataset
# データセットでXとYを検索する
# =============================================================================
all_col = dataset.columns.tolist()
ind_var = all_col[0:len(all_col)-1]

X = dataset[ind_var]
Y = dataset['charges'].values
# =============================================================================
# Split dataset
# データセットを分割する
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)
# =============================================================================
#  Call Algorithm
# 呼び出しアルゴリズム
# =============================================================================
import numpy as np
np.random.seed(0)
from sklearn.neural_network import MLPRegressor
#model = MLPRegressor( hidden_layer_sizes =(30,30,30))

# =============================================================================
# You can call other algorithms such as: KNN
# 次のような他のアルゴリズムを呼び出すことができます：KNN
#from sklearn.neighbors import KNeighborsRegressor
#model_knn = KNeighborsRegressor(n_neighbors=8)
##Linear regression
#from sklearn.linear_model import LinearRegression
#model_lg=LinearRegression()
##Decision Tree
#from sklearn.tree import DecisionTreeRegressor
#model = DecisionTreeRegressor(max_depth=12)
##SVC
#from sklearn.svm import SVR
#model = SVR()
##random forest
#from sklearn.ensemble import RandomForestRegressor
#np.random.seed(1337)
#model = RandomForestRegressor(n_estimators = 100,max_features= 5, random_state = 0)
##gradient boosting
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 2,
         'learning_rate': 0.01, 'loss': 'ls'}
model = GradientBoostingRegressor(**params)
# =============================================================================
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test, y_predict)
MSE = metrics.mean_squared_error(y_test, y_predict)
print('ANN-MAE: %.2f'%(MAE),'%')
print('ANN-MSE: %.2f'%(MSE),'%')

# =============================================================================
# HOLD-OUT TEST
# =============================================================================
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(solver ='lbfgs', hidden_layer_sizes =(30,30,30))
model.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score 
cv_scores = cross_val_score(model,X,Y, cv= 10, scoring = 'neg_mean_squared_error')
accuracy = abs(np.mean(cv_scores))
#print('Average of mean square error at 10 folds : %.2f'%(accuracy*100),'%')

# =============================================================================
# IMPORTANCE FEATURE
# =============================================================================
#import matplotlib.pyplot as plt
#
#plt.figure(figsize=(20,8))
#x = np.arange(0,40)
#y = y_test
#y2 =y_predict
#plt.plot(x, y)
#plt.plot(x, y2)
#plt.xlabel('x - axis')
#plt.ylabel('y - axis')
#plt.title('true value')
#plt.show()


#print("===============IMPORTANCE OF VARIABLE================")
#from sklearn.ensemble import RandomForestRegressor
#np.random.seed(1337)
#model = RandomForestRegressor(n_estimators = 100,max_features= 5, random_state = 0)
#model.fit(X_train,y_train)
#importance = model.feature_importances_
#for a, b in zip(ind_var,importance):
#    print(a,': ',round(100*b,2),'%')
#
#import matplotlib.pyplot as plt
#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(25).plot(kind='barh')
#plt.show()
