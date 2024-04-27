import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
# Basic Kernels
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
# More Kernels
from sklearn.gaussian_process.kernels import (RBF, 
                                              Matern, 
                                              DotProduct, 
                                              RationalQuadratic)

from modAL.models.learners import ActiveLearner
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
import joblib
# Our class
from sklearn.utils import shuffle


# For active learning, we shall define a custom query strategy 
# tailored to Gaussian processes (gp).
# Select the prediction y_gp that has the largest standard deviation 
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

# Function to split the data for training/testing    
def split (X,samples,val):
    X_train = X[:samples]
    X_val = X[samples:samples+val]
    X_test = X[samples+val:]
    return X_train,X_val,X_test
    
      
# =====================================
#           Main program
# =====================================
# Import data and shuffle it 
df = shuffle(pd.read_csv('data_all.csv'))

# define X and y
Xori = np.array(df.drop(['letters', 'disassoc', 'assoc'], axis=1))
y = np.array(df['assoc'])
# reshape y from vevtor to matrix
y = y.reshape(-1,1)

n_samples = 40
n_val = int(len(y)) - n_samples
X_train, X_val, X_test = split(Xori, n_samples, n_val)
y_train, y_val, y_test = split(y, n_samples, n_val)

# Number of samples for active learning 5,10,...
n_queries = np.linspace(5,n_samples,int(n_samples/5),dtype=int)

# Scaling 
# X
scaler_x = MinMaxScaler()
scaler_x.fit(X_train)
X_train_scale = scaler_x.transform(X_train)
scaler_x.fit(X_val)
X_val_scale = scaler_x.transform(X_val)
# y
scaler_y = MinMaxScaler()
scaler_y.fit(y_train)
y_train_scale = scaler_y.transform(y_train)
scaler_y.fit(y_val)
y_val_scale = scaler_y.transform(y_val)

# Defining Active learner
# Defining kernel for Gaussian Process Regressor 
kernel = DotProduct()
# Defining the active learnr using modAL package  
regressor = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel, 
                                                             random_state=0,
                                                             n_restarts_optimizer=0,
                                                             alpha=1),
                             query_strategy=GP_regression_std                              
                         )

# Function to iterate through the number of queries
def AL (n_queries):
    global query_indx, query_inst, y_pred_inv
    query_indx = []
    query_inst = []
    for idx in range(n_queries):
        query_idx, query_instance = regressor.query(X_train_scale)
        query_indx.append(query_idx)
        query_inst.append(query_instance)
        regressor.teach(X_train_scale[query_idx].reshape(1,-1), 
                        y_train_scale[query_idx].reshape(1,-1))
    # Trained model Prediction on unseen data
    global y_true, y_pred_kj
    y_pred = regressor.predict(X_val_scale, return_std=False)
    y_pred = y_pred.reshape(-1,1)
    y_pred_kj = scaler_y.inverse_transform(y_pred)
    # True data to calculate RMSE and R2
    y_true = y_val
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_kj))
    r2 = r2_score(y_true, y_pred_kj)
    # save the model 
    joblib.dump(regressor, "model.pkl") 
    return rmse,r2

# AL function implementation
rmse = []
for i in n_queries:
    rmse.append(AL(i))
    
# Plotting
res = np.array(rmse)
fig,ax1 = plt.subplots()
ax1.set_xlabel('Number of queries')
ax1.set_ylabel('RMSE kJ/mol', color='tab:red')
ax1.plot(n_queries,res[:,0], color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
plt.plot(n_queries,res[:,1], color='tab:blue')
ax2.set_ylabel('Coefficient of Determination (R$^2$)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
plt.savefig('modAL.png', dpi=500)

