import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your dataset and split it into features and target variable
X = np.loadtxt(r"C:\Users\SS Studios\Desktop\UMD_WINTER_2023\COURSES\CL\Mid project\project_1_files_CIS_581\final\train.dat", usecols=(0), unpack=True)
y = np.loadtxt(r"C:\Users\SS Studios\Desktop\UMD_WINTER_2023\COURSES\CL\Mid project\project_1_files_CIS_581\final\train.dat", usecols=(1), unpack=True)

X = X.reshape(-1,1)
y = y.reshape(-1,1)
#print(X,y)
# Define the range of polynomial degrees to test
degrees = np.arange(0, 13)

# Define the number of folds for cross-validation
n_folds = 6

# Initialize an empty list to store the mean squared errors for each degree
mse_scores = []

# Initialize a k-fold cross-validation object
kf = KFold(n_splits=n_folds, shuffle=False)

# Loop through the polynomial degrees and fit a polynomial regression model for each degree
for degree in degrees:
    # Initialize a polynomial regression model pipeline with scaling
    model = make_pipeline(StandardScaler(),PolynomialFeatures(degree, include_bias = True),StandardScaler(), LinearRegression())

    # Initialize an empty list to store the mean squared errors for each fold
    fold_scores = []

    # Loop through each fold in the k-fold cross-validation
    for train_idx, test_idx in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_test = np.array(X[train_idx]), np.array(X[test_idx])
        y_train, y_test = np.array(y[train_idx]), np.array(y[test_idx])
        
        
        # Fit the model on the training data for this fold
        model.fit(X_train, y_train)

        # Evaluate the model on the test data for this fold
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Add the mean squared error for this fold to the list of fold scores
        fold_scores.append(np.sqrt(mse))

    # Calculate the mean squared error for this degree by taking the average of the fold scores
    mse_degree = np.mean(fold_scores)

    # Add the mean squared error for this degree to the list of mse scores
    mse_scores.append(mse_degree)

# Print the degree with the lowest mean squared error
best_degree = np.argmin(mse_scores)
print(f"Best degree: {best_degree}, MSE: {mse_scores[best_degree]}")

