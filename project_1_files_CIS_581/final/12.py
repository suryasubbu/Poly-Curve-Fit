

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

## Reading the dat files
year_train, indicator_train = np.loadtxt("train.dat", usecols=(0,1), unpack=True)
year_test, indicator_test = np.loadtxt("test.dat", usecols=(0,1), unpack=True)

## preprocessing
x = year_train.reshape(-1,1)
y = indicator_train.reshape(-1,1)
X = year_test.reshape(-1,1)
Y = indicator_test.reshape(-1,1)

## polynomial and l2 penalizing parameters
lambda_val_ = [0,math.exp(-25),math.exp(-20),
             math.exp(-14),math.exp(-7),
             math.exp(-3),1,math.exp(7),
             math.exp(3)]
degree = np.arange(0,13)

scaler = preprocessing.StandardScaler().fit(x)
x_scaled = scaler.transform(x)
scaler = preprocessing.StandardScaler().fit(X)
X_scaled= scaler.transform(X)

d=6
poly = PolynomialFeatures(degree=d)
X__trans = poly.fit_transform(x_scaled)
X_test_trans = poly.fit_transform(X_scaled)
identity_matrix = np.identity(d+1)
w = (np.linalg.inv(X__trans.T @ X__trans + math.exp(-3) * identity_matrix) @ X__trans.T @ y)
y_hat1 = X_test_trans @ w

print(math.sqrt((np.square(y_hat1-Y).mean())))

