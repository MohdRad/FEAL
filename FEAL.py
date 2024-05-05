import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, 
                                              Matern, 
                                              DotProduct, 
                                              RationalQuadratic)

from modAL.models.learners import ActiveLearner
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
# Our class
import random
from DimensionReduction.DR import DRAL
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


# Function to define the ML model used and query strategy 
def query_str (q_str, est):
    regressor = ActiveLearner(estimator=est,
                             query_strategy=q_str)
    return regressor


# Function to plot the output of FE_AL function introduced down
def plotting (arr, arr_r, name):
    fig,ax1 = plt.subplots()
    ax1.set_xlabel('Number of queries')
    ax1.set_ylabel('RMSE kJ/mol', color='tab:red')
    ax1.plot(arr[:,0], arr[:,1], color='tab:red',label='Max. Std')
    ax1.plot(arr_r[:,0],arr_r[:,1], '--', color='tab:red', label='Random')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    plt.legend(loc='center right') # loc='center right'
    ax2 = ax1.twinx()
    plt.plot(arr[:,0], arr[:,2], color='tab:blue',label='Max. Uncertainty')
    plt.plot(arr_r[:,0], arr_r[:,2], '--', color='tab:blue', label='Random')
    ax2.set_ylabel('Coefficient of Determination (R$^2$)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.savefig(name, dpi=500, bbox_inches='tight')




# GPR training using Active Learning    
def FE_AL (df, n_samples, spacing, pca, shuf):
    '''
    Parameters
    ----------
    df :        (str) The name and directory of the data frame (.csv)
    n_samples : (int) Number of samples used in the active learning
    spacing :   (int) The increment in the number of samples make 
                      Make sure n_samples/spacing is an integer
    pca:        (str) choices are 'True' OR 'False', apply PCA 
    shuf:       (str) choices are 'True' OR 'False', apply data shuffle 

    Returns
    -------
    metrics: (array) an array with number of queries,Root Mean Squared Error, 
                     and cofficient of determination 

    '''
    # Read data csv    
    df = pd.read_csv(df)
    if (shuf=='True'):
        df = shuffle(df)
    # define X and y
    global Xori
    # X: Use PCA
    if (pca == 'yes'):
        obj = DRAL('./cases/PCA.csv')
        Xpca,Xori = obj.PCA(var=0.95)    
    else:
        Xori = np.array(df.drop(['letters', 'disassoc', 'assoc'], axis=1))
    # y
    y = np.array(df['assoc'])
    # reshape y from vevtor to matrix
    y = y.reshape(-1,1)
    # Number of samples for validation 
    n_val = int(len(y)) - n_samples
    X_train, X_val, X_test = split(Xori, n_samples, n_val)
    y_train, y_val, y_test = split(y, n_samples, n_val)
    # Number of samples for active learning 
    # Spacing 
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
    # Defining the active learner using modAL package 
    gpr = GaussianProcessRegressor(kernel=kernel,
                         random_state=0,
                         n_restarts_optimizer=0,
                         alpha=3)
    # Use GPR as an Active Learner, Max Std as a query strategy
    regressor = ActiveLearner(estimator=gpr,
                             query_strategy=GP_regression_std)
    
    # Empty lists to store the results
    metrics = []
    for k in range (0,len(n_queries)):
        n_query = n_queries[k]
        global query_indx, query_inst, y_pred_inv
        query_indx = []
        query_inst = []
        for idx in range(n_query):
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
        metrics.append([n_query,rmse,r2])
        # save the model 
        joblib.dump(regressor, "model.pkl") 
    return np.array(metrics)


# GPR training using random samples (Standard Training)
def rand (df,n_samples,spacing,pca,shuf):
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel,
                         random_state=0,
                         n_restarts_optimizer=0,
                         alpha=3)
    # Read data csv    
    df = pd.read_csv(df)
    if (shuf=='True'):
        df = shuffle(df)
    # define X and y
    if (pca == 'True'):
        obj = DRAL('./cases/PCA.csv')
        Xpca,Xori = obj.PCA(var=0.95)  
    else:
        Xori = np.array(df.drop(['letters', 'disassoc', 'assoc'], axis=1))
           
    y = np.array(df['assoc'])
    # reshape y from vevtor to matrix
    y = y.reshape(-1,1)
    # Number of samples for validation 
    n_val = int(len(y)) - n_samples
    X_train, X_val, X_test = split(Xori, n_samples, n_val)
    y_train, y_val, y_test = split(y, n_samples, n_val)
    # Number of samples for active learning 
    # Spacing 
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
    metrics = []
    for n_samples in n_queries:
        gpr.fit(X_train_scale[0:n_samples], y_train_scale[0:n_samples])
        y_pred = gpr.predict(X_val_scale)
        y_pred = y_pred.reshape(-1,1)
        y_pred_kj = scaler_y.inverse_transform(y_pred)
        # True data to calculate RMSE and R2
        y_true = y_val
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_kj))
        r2 = r2_score(y_true, y_pred_kj)
        metrics.append([n_samples,rmse,r2])
    return np.array(metrics)     
    


