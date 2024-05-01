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
    ax1.plot(arr_r[:,0],arr_r[:,1], '--', color='tab:red', label='Random Std')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    plt.legend(loc='center right') # loc='center right'
    ax2 = ax1.twinx()
    plt.plot(arr[:,0], arr[:,2], color='tab:blue',label='Max. Uncertainty')
    plt.plot(arr_r[:,0], arr_r[:,2], '--', color='tab:blue', label='Random')
    ax2.set_ylabel('Coefficient of Determination (R$^2$)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.savefig(name, dpi=500, bbox_inches='tight')
    
def FE_AL (df, n_samples, spacing, q_str):
    '''
    Parameters
    ----------
    df : (str) The name and directory of the data frame (.csv)
    n_samples : (int) Number of samples used in the active learning
    spacing : (int) The increment in the number of samples make 
                    Make sure n_samples/spacing is an integer
    q_str : (str) The choices are (1) max or (2) random
                  max: select the prediction with maximum standard devation  
                  random: select the prediction randomly 

    Returns
    -------
    metrics: (array) an array with number of queries,Root Mean Squared Error, 
                     and cofficient of determination 

    '''
    # Read data csv    
    df = pd.read_csv(df)
    # define X and y
    Xori = np.array(df.drop(['letters', 'disassoc', 'assoc'], axis=1))
    y = np.array(df['assoc'])
    # reshape y from vevtor to matrix
    y = y.reshape(-1,1)
    #n_samples = 30
    n_val = int(len(y)) - n_samples
    X_train, X_val, X_test = split(Xori, n_samples, n_val)
    y_train, y_val, y_test = split(y, n_samples, n_val)
    # Number of samples for active learning 5,10,...
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
    # Use function query_str to define the regressor and query strategy
    if (q_str == "max"):
            regressor = query_str(GP_regression_std,gpr)
    elif (q_str == "random"):
           regressor = query_str(rand_std,gpr)
    
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



    



