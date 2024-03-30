# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:58:26 2023

@author: SS Studios
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
import warnings
warnings.filterwarnings("ignore")


year_train = np.loadtxt(r"train.dat", usecols=(0), unpack=True)
indicator_train = np.loadtxt(r"train.dat", usecols=(1), unpack=True)
year_test, indicator_test = np.loadtxt(r"test.dat", usecols=(0,1), unpack=True)

X = year_train.reshape(-1,1)
y = indicator_train.reshape(-1,1)
X_test = year_test.reshape(-1,1)
Y_test = indicator_test.reshape(-1,1)

def best_degree(X,y):
    degrees = np.arange(1, 13)

    # Define the number of folds for cross-validation
    n_folds = 6

    # Initialize an empty list to store the mean squared errors for each degree
    rmse_scores = []

    # Initialize a k-fold cross-validation object
    kf = KFold(n_splits=n_folds, shuffle=False)

    # Loop through the polynomial degrees and fit a polynomial regression model for each degree
    for degree in degrees:
        # Initialize a polynomial regression model pipeline with scaling
        model = make_pipeline(PolynomialFeatures(degree, include_bias = False), LinearRegression())
        s = StandardScaler()
        # Initialize an empty list to store the mean squared errors for each fold
        fold_scores = []
        X = (X-np.mean(X))/np.std(X)
        # Loop through each fold in the k-fold cross-validation
        for train_idx, test_idx in kf.split(X):
            # Split the data into training and test sets for this fold
            X_train, X_test = np.array(X[train_idx]), np.array(X[test_idx])
            y_train, y_test = np.array(y[train_idx]), np.array(y[test_idx])
            
            
            
            # Fit the model on the training data for this fold
            model.fit(X_train, y_train)

            # Evaluate the model on the test data for this fold
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Add the mean squared error for this fold to the list of fold scores
            fold_scores.append(rmse)

        # Calculate the mean squared error for this degree by taking the average of the fold scores
        rmse_degree = np.mean(fold_scores)

        # Add the mean squared error for this degree to the list of mse scores
        rmse_scores.append([degree,rmse_degree])
        
    degree_df = pd.DataFrame(rmse_scores)
    degree_df.columns = ["d","avg RMSE"]
    
    plt.plot(degree_df.d , degree_df["avg RMSE"])
    plt.xlabel("Degree")
    plt.ylabel("AVERAGE RMSE")
    plt.title("Error curve for linear regression (Lambda = 0)")
    plt.show()
    print("best_degree : {} and its RMSE {}".format(degree_df[degree_df["avg RMSE"] == degree_df["avg RMSE"].min()].d[5],degree_df["avg RMSE"].min()))
    print("")
    return degree_df

def best_lambda(X,y):
    lambda_ = [0,math.exp(-25),math.exp(-20),
                 math.exp(-14),math.exp(-7),
                 math.exp(-3),1,math.exp(3),
                 math.exp(7)]
    
    # Define the number of folds for cross-validation
    n_folds = 6
    s = StandardScaler()
    # Initialize an empty list to store the mean squared errors for each degree
    rmse_scores = []
    
    # Initialize a k-fold cross-validation object
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    # Loop through the polynomial degrees and fit a polynomial regression model for each degree
    for l in lambda_:
        # Initialize a polynomial regression model pipeline with scaling
        model = make_pipeline(PolynomialFeatures(degree = 12, include_bias = False), Ridge(alpha=l,fit_intercept=True,solver='cholesky'))
    
        # Initialize an empty list to store the mean squared errors for each fold
        fold_scores = []
        X = (X-np.mean(X))/np.std(X)
        # Loop through each fold in the k-fold cross-validation
        for train_idx, test_idx in kf.split(X):
            # Split the data into training and test sets for this fold
            X_train, X_test = np.array(X[train_idx]), np.array(X[test_idx])
            y_train, y_test = np.array(y[train_idx]), np.array(y[test_idx])
            
            # Fit the model on the training data for this fold
            model.fit(X_train, y_train)
    
            # Evaluate the model on the test data for this fold
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
            # Add the mean squared error for this fold to the list of fold scores
            fold_scores.append(rmse)
    
        # Calculate the mean squared error for this degree by taking the average of the fold scores
        rmse_degree = np.mean(fold_scores)
    
        # Add the mean squared error for this degree to the list of mse scores
        rmse_scores.append([l,rmse_degree])
        
    degree_df = pd.DataFrame(rmse_scores)
    degree_df.columns = ["l","avg RMSE"]
    plt.plot(degree_df.l , degree_df["avg RMSE"])
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("AVERAGE RMSE")
    plt.title("Error curve for linear regression (degree = 12)")
    plt.show()
    print("best_lambda(regularization_factor) : {} and its RMSE {}".format(degree_df[degree_df["avg RMSE"] == degree_df["avg RMSE"].min()].l[5],degree_df["avg RMSE"].min()))
    
    print("")
    return degree_df


def entire_training_rmse(X,y,degree,l):
    X = (X - np.mean(X))/np.std(X)
    regr = make_pipeline(PolynomialFeatures(degree, include_bias = False), Ridge(alpha=l,fit_intercept=True,solver='cholesky'))
    regr.fit(X, y)
    w = regr.named_steps['ridge'].coef_
    # Evaluate the model on the test data for this fold
    y_pred = regr.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))


    print("Training RMSE for degree {} and lambda{} is".format(degree,l),rmse)
    print("")
    return w,rmse

def entire_test_rmse(X,X_test,y,y_test,degree,l):
    X_test = (X_test - np.mean(X))/np.std(X)
    X = (X - np.mean(X))/np.std(X)
    
    regr = make_pipeline(PolynomialFeatures(degree, include_bias = False), Ridge(alpha=l,fit_intercept=True,solver='cholesky'))
    regr.fit(X, y)
    # Evaluate the model on the test data for this fold
    y_pred = regr.predict(X)
    y_pred_1 = regr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred_1, y_test))


    xnew = np.linspace(X.min(), X.max(), 300) 
    df = pd.DataFrame(np.array(X))
    df.columns = ["X"]
    df["y_pred"] = y_pred
    df.sort_values("X",inplace = True)
    spl = make_interp_spline(df.X, df.y_pred, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    sns.scatterplot( X.flatten(),y.flatten())
    sns.lineplot(xnew,power_smooth)
    #sns.scatterplot( X_test.flatten(),y_pred_1.flatten())
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    print("Testing RMSE for degree {} and lambda{} is".format(degree,l),rmse)
    print("")
    return rmse
    
b_d = best_degree(X,y)
b_l = best_lambda(X, y)
w_t_6,rmse_t_6 = entire_training_rmse(X,y,6,np.exp(-3))
rmse_test_6 = entire_test_rmse(X,X_test,y,Y_test,6,np.exp(-3))
w_t_12,rmse_t_12 = entire_training_rmse(X,y,12,np.exp(-3))
rmse_test_12 = entire_test_rmse(X,X_test,y,Y_test,12,np.exp(-3))
w_t_6,rmse_t_6 = entire_training_rmse(X,y,6,0)
rmse_test_6 = entire_test_rmse(X,X_test,y,Y_test,6,0)
w_t_12,rmse_t_12 = entire_training_rmse(X,y,12,0)
rmse_test_12 = entire_test_rmse(X,X_test,y,Y_test,12,0)