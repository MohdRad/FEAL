import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
import random


# For active learning, we shall define a custom query strategy 
# tailored to Gaussian processes (gp).
# Select the prediction y_gp that has the largest standard deviation 
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]
# Select the samples based on random standard deviation 
def rand_std(regressor,X):
    mean, std = regressor.predict(X, return_std=True)
    a = random.choice(std)
    b=np.where(std==a)[0]
    c = b[0]
    return c, X[c]

# Function that randomly select from the model prediction
# This is compatible with the other algorithms 
def rand_sel(regressor,X):
    pred = regressor.predict(X)
    a = random.choice(pred)
    b=np.where(pred==a)[0]
    c = b[0]
    return c, X[c]

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
df = pd.read_csv('data_all.csv')

# define X and y
Xori = np.array(df.drop(['letters', 'disassoc', 'assoc'], axis=1))
y = np.array(df['assoc'])
# reshape y from vevtor to matrix
y = y.reshape(-1,1)

n_samples = 30
n_val = int(len(y)) - n_samples
X_train, X_val, X_test = split(Xori, n_samples, n_val)
y_train, y_val, y_test = split(y, n_samples, n_val)

# Number of samples for active learning 5,10,...
spacing = 2
n_queries = np.linspace(spacing,n_samples,int(n_samples/spacing),dtype=int)

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
# Function is to change the query strategy 
gpr = GaussianProcessRegressor(kernel=kernel,
                         random_state=0,
                         n_restarts_optimizer=0,
                         alpha=3)

# Function to define the ML model used and query strategy 
def query_str (q_str, est):
    regressor = ActiveLearner(estimator=est,
                             query_strategy=q_str)
    return regressor

# GPR based on maximum Standard deviation 
regressor_gpr = query_str(GP_regression_std,gpr)
# GPR based on random standard deviation 
regressor_rand = query_str(rand_std, gpr)
# GBR based on random prediction 




# Function to iterate through the number of queries
def AL (n_queries, regressor):
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
    y_pred = regressor.predict(X_val_scale)
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

# AL function implementation, store RMSE and R2 into lists 
rmse = []
rmse_rand = []
for i in n_queries:
    rmse.append(AL(i,regressor_gpr))
    rmse_rand.append(AL(i,regressor_rand))

# Plotting
res = np.array(rmse)
res_rand = np.array(rmse_rand)
fig,ax1 = plt.subplots()
ax1.set_xlabel('Number of queries')
ax1.set_ylabel('RMSE kJ/mol', color='tab:red')
ax1.plot(n_queries,res[:,0], color='tab:red',label='Max. Std')
ax1.plot(n_queries,res_rand[:,0], '--', color='tab:red', label='Random Std')
ax1.tick_params(axis='y', labelcolor='tab:red')
plt.legend(loc='center right') # loc='center right'
ax2 = ax1.twinx()
plt.plot(n_queries,res[:,1], color='tab:blue',label='Max. Uncertainty')
plt.plot(n_queries,res_rand[:,1], '--', color='tab:blue', label='Random')
ax2.set_ylabel('Coefficient of Determination (R$^2$)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
plt.savefig('modAL.png', dpi=500, bbox_inches='tight')

