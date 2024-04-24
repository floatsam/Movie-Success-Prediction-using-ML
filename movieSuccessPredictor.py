import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#import scipy.stats
import math as mp

#Reading movieDetails.csv dataset
dataSet = pd.read_csv('C:\\Users\\anird\\OneDrive\\Desktop\\Uni Books\\Semester VI\\INT354\Project\\movieDetails.csv', encoding = 'unicode_escape')
dataSet.info()

#Data Pre Processing
#Cleaning the dataset
dataSet = dataSet.drop(['MovieID'], axis = 1)
dataSet['Title'] = dataSet['Title'].astype('string')
dataSet['MPAA_Rating'] = dataSet['MPAA_Rating'].astype('string')
dataSet['Budget'] = dataSet['Budget'].astype('string').str.replace('$', '').str.replace(',', '').astype('float')
dataSet['Gross'] = dataSet['Gross'].astype('string').str.replace('$', '').str.replace(',', '').astype('float') #these two featuresets contain a lot of nan values, let's fix it!
dataSet['Budget'] = dataSet['Budget'].fillna(dataSet['Budget'].mean())
dataSet['Gross'] = dataSet['Gross'].fillna(dataSet['Gross'].mean())
dataSet.drop('Release_Date', axis = 1, inplace = True) #Release date has no impact on the movie performance in this dataset...
dataSet['Rating'] = dataSet['Rating'].fillna(dataSet['Rating'].median())#in this case, median data can be inserted in NaN fields to balance out this featureset better as compared to using mean data, as that can increase the ratings and skew success metrics more that it need to be!!!
for i in dataSet['Rating'] : print(i)
dataSet['Rating Count'] = dataSet['Rating Count'].str.replace(',', '').astype('string')
dataSet['Rating Count'] = dataSet['Rating Count'].astype('float')
dataSet['Rating Count'] = dataSet['Rating Count'].fillna(dataSet['Rating Count'].mean())
dataSet['Genre'] = dataSet['Genre'].astype('string')
dataSet.drop('Summary', axis = 1, inplace = True) #Summary adds nothing constructive to our dataset, so we drop it!
dataSet.info()

#Visualizing the relationship between features to check possible relation and variances to pick the best ML models for them before we do feature engineering...

#1. Ratings vs Revenue graph
plt.scatter(x = dataSet['Rating'],y = dataSet['Gross'])
plt.title("Finding linear relationship between Ratings and revenue of movies")
plt.xlabel("Rating")
plt.ylabel("Gross")
plt.show()#We can see high overlap between features...

#We will consider Gross Revenue generated to be the target, the independent variable which determines the success of the movie from past metrics
#We will check for the distribution of Gross using a histogram
sns.histplot(dataSet['Gross'], kde=True, bins=30, color='green')
plt.show()

#2. Heatmap B/W Budget & Gross
temp = dataSet.copy()
temp = temp.drop(['MPAA_Rating'], axis = 1)
temp = temp.drop(['Title'], axis = 1)
temp = temp.drop(['Genre'], axis = 1)
#temp = temp.drop([])
sns.heatmap(temp.corr(), annot = True)
plt.show()

#Feature engineering i:e One hot encoding, extracting relevant datasets etc
#Here, two featuresets (MPAA_Rating & Genre can be OHE to convert them into distinct binary featuresets of n(unique feature) values)

featureSet = dataSet.copy()
featureSet = featureSet.drop(['Title'], axis = 1)
featureSetEncoded = pd.get_dummies(featureSet, columns = ['MPAA_Rating', 'Genre'])
featureSetEncoded.info()
#It's done!

#Now onto splitting test and train datasets
X = featureSetEncoded
X = X.drop(['Gross'], axis = 1)
X = X.values
Y = featureSetEncoded.iloc[ :, 1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=52)

#Firstly we will try a simple algorithm, Random Forest Regressor and then check for accuracy
#Doing Hyperparameter tuning by repeatedly changing parameters of various algorithms, esentially a lot of trial and error
RndForest = RandomForestRegressor(n_estimators = 300)
RndForest.fit(X_train, Y_train)
RndForestPrediction = RndForest.predict(X_test)
r2_RndFor = r2_score(Y_test, RndForestPrediction) #R2_score, the most reliable metric to find accuracy of the model
rms_RndFor = mp.sqrt(mean_squared_error(Y_test, RndForestPrediction, squared=False))
acc_RndForest = RndForest.score(X_test, Y_test)

print("R2 score for this regression model is ", r2_RndFor, " points!")
print("Root Mean Square Error for this regression model is: ", rms_RndFor)
print("Accuracy : ", acc_RndForest * 100)

#Secondly, we will try a more effective algorithm, lets do it....
#We will use voting regressor which will pass the same dataset through multiple regression models and the modal outcome of these models will be considered as the right outcome, essentially....
RndForestEst = RandomForestRegressor(n_estimators = 665)
LinearEst = LinearRegression()
SvR_Est = SVR(kernel = 'sigmoid')
VotingReg = VotingRegressor(estimators=[('linear', LinearEst),  ('svr', SvR_Est), ('rf', RndForestEst)], n_jobs=6)
VotingReg.fit(X_train, Y_train)
VotingRegressorPrediction = VotingReg.predict(X_test)
r2_VotReg = r2_score(Y_test, VotingRegressorPrediction)
rms_VotReg = mp.sqrt(mean_squared_error(Y_test, VotingRegressorPrediction, squared=False))
acc_VotingReg = VotingReg.score(X_test, Y_test)

print("R2 score for this regression model is ", r2_VotReg, " points!")
#It seems this algorithm is giving us worse scores, so this model cannot explain the variance as well as the model given above...
print("Root Mean Square Error for this regression model is: ", rms_VotReg)
print("Accuracy : ", acc_VotingReg * 100)

#Third algorithm we will use, is from different package altogether, it is part of xgboost package, and it is a boosted ensemble learning algorithm called XGBoost which uses the concept of gradient boosting
XGB_Est = XGBRegressor(n_estimators = 650, max_depth = 8, eta = 0.03, subsample = 0.8, colsample_bytree = 0.8)
XGB_Est.fit(X_train, Y_train)
XGB_pred = XGB_Est.predict(X_test)
r2_XGBReg = r2_score(Y_test, XGB_pred)
rms_XGBReg = mp.sqrt(mean_squared_error(Y_test, XGB_pred, squared=False))
acc_XGBRegressor = XGB_Est.score(X_test, Y_test)

print("R2 score for this regression model is ", r2_XGBReg, " points!")
print("Root Mean Squared Error for this regression model is ", rms_XGBReg)
print("Accuracy : ", acc_XGBRegressor * 100)

#Fourth algorithm we will use, is called catboost algorithm, which is a gradient boosting alg. which combines and operates over several weak learners into a strong learner in order to minimize mean squared error to a large extent
CatBoost_Est = CatBoostRegressor(iterations = 350, learning_rate = 0.03)
CatBoost_Est.fit(X_train, Y_train, eval_set = (X_test, Y_test))
CatBoost_pred = CatBoost_Est.predict(X_test)
r2_CatBoostReg = r2_score(Y_test, CatBoost_pred)
rms_CatBoostReg = mp.sqrt(mean_squared_error(Y_test, CatBoost_pred, squared=False))
acc_CatBoostReg = CatBoost_Est.score(X_test, Y_test)

print("R2 score for this regression model is ", r2_CatBoostReg, " points!")
print("Root Mean Squared Error for this regression model is ", rms_CatBoostReg)
print("Accuracy : ", acc_CatBoostReg * 100)

#So, conclusion is that CatBoost and Ensemble Forest algorithms give us the best results as evident by their r2 score and rmse, which is correct as
#the dataset is highly dimensional and features complex non linear relationships and high overlapping as evident by eda performed on the dataset previously