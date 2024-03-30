

## packages to handle array and datafram
import numpy as np
import pandas as pd

## Sklearn packages
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

def scaling_transforming_scaling(x_,d):
    #mean = np.array(x_).mean()
    #std = np.array(x_).std
    #x_scal = (x_ - mean)/std()
    poly = PolynomialFeatures(degree=d, include_bias = True)
    X_train_poly = poly.fit_transform(x_)
    #mean_t = np.array(X_train_poly).mean()
    #std_t = np.array(X_train_poly).std()
    #ans = (X_train_poly - mean_t)/std_t
    #print(np.array(ans).shape)
    #print(np.array(X_train_poly))
    return X_train_poly
    
## Reading the dat files
year_train, indicator_train = np.loadtxt(r"C:\Users\SS Studios\Desktop\UMD_WINTER_2023\COURSES\CL\Mid project\project_1_files_CIS_581\final\train.dat", usecols=(0,1), unpack=True)
year_test, indicator_test = np.loadtxt(r"C:\Users\SS Studios\Desktop\UMD_WINTER_2023\COURSES\CL\Mid project\project_1_files_CIS_581\final\test.dat", usecols=(0,1), unpack=True)


## preprocessing
x_train = year_train.reshape(-1,1)
y_train = indicator_train.reshape(-1,1)
X_test = year_test.reshape(-1,1)
Y_test = indicator_test.reshape(-1,1)

x_train = (x_train - np.mean(x_train))/np.std(x_train)
### Training sets
xtrain_1 = np.array(x_train[7:42])
xtrain_2 = np.concatenate((x_train[0:7],x_train[14:42]))
xtrain_3 = np.concatenate((x_train[0:14],x_train[21:42]))
xtrain_4 = np.concatenate((x_train[0:21],x_train[28:42]))
xtrain_5 = np.concatenate((x_train[0:28],x_train[35:42]))
xtrain_6 = np.array(x_train[0:35])
xcv_train = np.array([xtrain_1,xtrain_2,xtrain_3,xtrain_4,xtrain_5,xtrain_6])
del(xtrain_1,xtrain_2,xtrain_3,xtrain_4,xtrain_5,xtrain_6)
ytrain_1 = np.array(y_train[7:42])
ytrain_2 = np.concatenate((y_train[0:7],y_train[14:42]))
ytrain_3 = np.concatenate((y_train[0:14],y_train[21:42]))
ytrain_4 = np.concatenate((y_train[0:21],y_train[28:42]))
ytrain_5 = np.concatenate((y_train[0:28],y_train[35:42]))
ytrain_6 = np.array(y_train[0:35])
ycv_train = np.array([ytrain_1,ytrain_2,ytrain_3,ytrain_4,ytrain_5,ytrain_6])
del(ytrain_1,ytrain_2,ytrain_3,ytrain_4,ytrain_5,ytrain_6)
### test sets

xtest_1 = x_train[0:7]
xtest_2 = x_train[7:14]
xtest_3 = x_train[14:21]
xtest_4 = x_train[21:28]
xtest_5 = x_train[28:35]
xtest_6 = x_train[35:42]
xcv_test = np.array([xtest_1,xtest_2,xtest_3,xtest_4,xtest_5,xtest_6])
del(xtest_1,xtest_2,xtest_3,xtest_4,xtest_5,xtest_6)
ytest_1 = y_train[0:7]
ytest_2 = y_train[7:14]
ytest_3 = y_train[14:21]
ytest_4 = y_train[21:28]
ytest_5 = y_train[28:35]
ytest_6 = y_train[35:42]
ycv_test = np.array([ytest_1,ytest_2,ytest_3,ytest_4,ytest_5,ytest_6])
del(ytest_1,ytest_2,ytest_3,ytest_4,ytest_5,ytest_6)
# d is degree
weights = []
rmse_avg = []
lambda_range = [0,math.exp(-25),math.exp(-20),
             math.exp(-14),math.exp(-7),
             math.exp(-3),1,math.exp(7),
             math.exp(3)]
for d in range(0,13):
    w_ = []
    rmse = []
    for i in range(6):
        alpha = 0
        train_x = xcv_train[i]
        train_y = np.array(ycv_train[i])
        test_x = xcv_test[i]
        test_y = ycv_test[i]
        
        trans_train_x = scaling_transforming_scaling(train_x,d)
        trans_test_x = scaling_transforming_scaling(test_x,d)
        
        train_y_scale = (train_y - train_y.mean())/train_y.std()
        
        regr = Ridge(alpha=alpha,fit_intercept=True,solver='cholesky')
        regr.fit(trans_train_x,train_y_scale)
        w = regr.coef_
        y_hat = trans_test_x @ w[0]
        y_denormalize = (y_hat * train_y.std()) + train_y.mean()
        
        ## RMSE CALCULATION 
        RM = np.sqrt(np.mean(np.square(y_denormalize-test_y)))
        rmse.append(RM)
        w_.append([d,y_denormalize,regr.coef_])
    rmse_avg.append([d,np.array(rmse).mean()])
        
    weights.append(w_)
print(rmse_avg)    
    