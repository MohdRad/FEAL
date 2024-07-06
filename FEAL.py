import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, 
                                              Matern, 
                                              DotProduct, 
                                              RationalQuadratic,
                                              ExpSineSquared,
                                              ConstantKernel,
                                              WhiteKernel,
                                              Exponentiation,
                                              Sum,
                                              Product)

from modAL.models.learners import ActiveLearner, CommitteeRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
# Our class
from DimensionReduction.DR import DRAL
from modAL.disagreement import max_std_sampling
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import train_test_split
import shap


# For active learning, we shall define a custom query strategy 
# tailored to Gaussian processes (gp).
# Select the prediction y_gp that has the largest standard deviation 
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

# Function for Random sample selection 
def rand(domain):
    new_idx = np.random.choice(domain, size=1, replace=False)
    return new_idx


# Function to make curves smoother
def smoother (x,y):
    spl = make_interp_spline(x,y,k=3)
    x_new = np.linspace(min(x),max(x),300)
    y_smooth = spl(x_new)
    return x_new, y_smooth

  

# Function to plot the output of FE_AL function introduced down
def plotting (arr, arr_r, name, unc, 
              title,
              xrange, 
              xticks,
              y1_range,
              y1_ticks,
              y2_range,
              y2_ticks,
              zoom_coor,
              rmse_range, 
              rmse_ticks,  
              r2_range, 
              r2_ticks,
              legend_loc):
    '''
    Parameters
    ----------
    arr:    (arr) metrics of AL at n_samples
    arr_r:  (arr) metrics of random at n_samples
    name:   (str) name of output plot file 
    unc:    (bool) choices are True or False, to include uncertainty
  
    Returns
    -------
    saved plot in ./figs

    '''
    plt.rcParams.update({'font.size': 14})
    n_new, rmse_smooth = smoother(arr[:,0], arr[:,1])
    n_new, r2_smooth = smoother(arr[:,0], arr[:,2])
    n_new, rmse_r_smooth = smoother(arr_r[:,0], arr_r[:,1])
    n_new, r2_r_smooth = smoother(arr_r[:,0], arr_r[:,2])
    fig,ax1 = plt.subplots()
    plt.title(title)
    ax1.set_xlabel('Number of queries')
    ax1.set_ylabel('RMSE kJ/mol', color='tab:red')
    ax1.set_xlim([0,max(arr[:,0])+1.0])
    ax1.set_ylim(y1_range)
    ax1.set_yticks(y1_ticks)
    ax1.plot(n_new, rmse_smooth, color='tab:red',label='AL')
    ax1.plot(n_new,rmse_r_smooth, '--', color='tab:red', label='Random')
    plt.legend(loc='best', bbox_to_anchor=legend_loc)
    
    if (unc):
        n_new, rmse_unc_smooth = smoother(arr[:,0],arr[:,3])
        n_new, rmse_r_unc_smooth = smoother(arr_r[:,0],arr_r[:,3])
        plt.fill_between(n_new, rmse_smooth-rmse_unc_smooth, rmse_smooth+rmse_unc_smooth,
                         color='red', alpha=0.3)
        plt.fill_between(n_new, rmse_r_smooth-rmse_r_unc_smooth, rmse_r_smooth+rmse_r_unc_smooth,
                         color='grey', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    plt.legend(loc='best', bbox_to_anchor=legend_loc) 

   
    ax2 = ax1.twinx()
    plt.plot(n_new, r2_smooth, color='tab:blue',label='Max. Uncertainty')
    plt.plot(n_new, r2_r_smooth, '--', color='tab:blue', label='Random')
    ax2.set_ylim(y2_range)
    ax2.set_yticks(y2_ticks)
    
    if (unc):
        n_new, r2_unc_smooth = smoother(arr[:,0],arr[:,4])
        n_new, r2_r_unc_smooth = smoother(arr_r[:,0],arr_r[:,4])
        plt.fill_between(n_new, r2_smooth-r2_unc_smooth, r2_smooth+r2_unc_smooth,
                         color='blue', alpha=0.3)
        plt.fill_between(n_new, r2_r_smooth-r2_r_unc_smooth, r2_r_smooth+r2_r_unc_smooth,
                         color='grey', alpha=0.5)
    ax2.set_ylabel('R$^2$', 
                   color='tab:blue',
                   )
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    a = plt.axes(zoom_coor)
    plt.rcParams.update({'font.size': 10})
    plt.plot(n_new, rmse_smooth, color='tab:red')
    plt.plot(n_new,rmse_r_smooth, '--', color='tab:red')
    plt.xlim(xrange)
    plt.ylim(rmse_range)
    plt.xticks(xticks, fontsize=10)
    plt.yticks(rmse_ticks, fontsize=10, color='tab:red')
    if (unc):
        n_new, rmse_unc_smooth = smoother(arr[:,0],arr[:,3])
        n_new, rmse_r_unc_smooth = smoother(arr_r[:,0],arr_r[:,3])
        plt.fill_between(n_new, rmse_smooth-rmse_unc_smooth, rmse_smooth+rmse_unc_smooth,
                         color='red', alpha=0.3)
        plt.fill_between(n_new, rmse_r_smooth-rmse_r_unc_smooth, rmse_r_smooth+rmse_r_unc_smooth,
                         color='grey', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    a2 = a.twinx()
    plt.plot(n_new, r2_smooth, color='tab:blue')
    plt.plot(n_new, r2_r_smooth, '--', color='tab:blue')
    plt.ylim(r2_range)
    plt.yticks(r2_ticks, fontsize=10, color='tab:blue')
    if (unc):
        n_new, r2_unc_smooth = smoother(arr[:,0],arr[:,4])
        n_new, r2_r_unc_smooth = smoother(arr_r[:,0],arr_r[:,4])
        plt.fill_between(n_new, r2_smooth-r2_unc_smooth, r2_smooth+r2_unc_smooth,
                         color='blue', alpha=0.3)
        plt.fill_between(n_new, r2_r_smooth-r2_r_unc_smooth, r2_r_smooth+r2_r_unc_smooth,
                         color='grey', alpha=0.5)
    
    plt.savefig('./figs/'+name+'.png', dpi=500, bbox_inches='tight')
    
    
# GPR training using Active Learning    
def FE_AL (df, n_samples, seed, no_shuf, pca, fe, use_shap):
    '''
    Parameters
    ----------
    df :        (str)  The name and directory of the data frame (.csv)
    n_samples : (int)  Number of samples used in the active learning
    seed:       (int)  The way the data will be shuffled
    pca:        (bool) choices are 'True' OR 'False', apply PCA 
    fe:         (str)  free energy to be used as an output, choices are 'assoc', 'disassoc'   
    Returns
    -------
    metrics: (array) an array with number of queries,Root Mean Squared Error, 
                     and cofficient of determination 

    '''
    np.random.seed(seed)
    # Read data csv    
    df = pd.read_csv(df)
    # define X and y
    X = df.drop(['letters','assoc','disassoc'], axis=1)
    y = df[fe]
    letters = np.array(df['letters'])
    if (pca):
        X.to_csv('./cases/PCA.csv', index=False)
        obj = DRAL(path_X='./cases/PCA.csv')
        X = obj.PCA(var=0.95)

    global X_test
    if (no_shuf):
        X_sample, X_test, y_sample, y_test, pac_sample, pac_test = train_test_split(X, y, letters, 
                                                                                    test_size=0.29,
                                                                                    random_state=None,
                                                                                    shuffle=False)

    else: 
    
       X_sample, X_test, y_sample, y_test, pac_sample, pac_test = train_test_split(X, y, letters, 
                                                                                   test_size=0.29,
                                                                                   random_state=seed)

    #if (use_shap):
    #    X_sample = X_sample.drop(['T'], axis=1)
    #    X_test = X_test.drop(['T'], axis=1)
    #    X = X.drop(['T'], axis=1)
    # reshape y from vevtor to matrix
    y_sample = np.array(y_sample).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    # Scaling 
    # X
    y = np.array(df[fe]).reshape(-1,1)
    scaler_x = MinMaxScaler()
    scaler_x.fit(X)
    X_sample_scale = scaler_x.transform(X_sample)
    X_test_scale = scaler_x.transform(X_test)
    # y
    scaler_y = MinMaxScaler()
    scaler_y.fit(y)
    y_sample_scale = scaler_y.transform(y_sample)
    # Defining Active learner
    # Defining kernel for Gaussian Process Regressor 
    kernel = DotProduct()
    # Defining the active learner using modAL package 
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                         random_state=0,
                         n_restarts_optimizer=0,
                         alpha=1)

                     
    # Start with n random samples
    idx_in=np.random.choice(len(X_sample_scale), size=1, replace=False)
    X_ini = X_sample_scale[idx_in]
    y_ini = y_sample_scale[idx_in]
    X_sample = np.delete(X_sample_scale, idx_in, axis=0)
    y_sample = np.delete(y_sample_scale, idx_in, axis=0) 
    X_sample_in = X_sample.copy()
    y_sample_in = y_sample.copy()
                    
    # Use GPR as an Active Learner, Max Std as a query strategy
    regressor = ActiveLearner(estimator=gpr,
                             query_strategy=GP_regression_std,
                             X_training=X_ini,
                             y_training=y_ini)
    
    # To calculate metrics every 5 points
    n_metrics = np.arange(0,n_samples+10,10)
    # Empty lists to store the results
    metrics = []
    pac_used = []
    k = 0
    for i in range (n_samples+1):
        query_idx, query_instance = regressor.query(X_sample)
        regressor.teach(X_sample[query_idx].reshape(1,-1), 
                        y_sample[query_idx].reshape(-1,1))
        pac_used.append([i,pac_sample[query_idx]])
        # Delete the query from the samples space to avoid reselection 
        X_sample = np.delete(X_sample, query_idx, axis=0)
        y_sample = np.delete(y_sample, query_idx, axis=0)
        # Metrics every 5 samples 
        if (i == n_metrics[k]):
            # Trained model Prediction on unseen data
            y_pred = regressor.predict(X_test_scale)
            y_pred = y_pred.reshape(-1,1)
            y_pred_kj = scaler_y.inverse_transform(y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
            r2 = r2_score(y_test, y_pred_kj)
            metrics.append([i,rmse,r2])
            k=k+1
    # save the model 
    joblib.dump(regressor, "./trained_models/"+fe+str(n_samples)+".pkl")
    
    if(use_shap):
        explainer = shap.KernelExplainer(model = regressor.predict, data = X_test)
        shap_obj = explainer(X_test)
        plt.rcParams['font.size'] = 18
        f1 = plt.figure()
        ax1 = f1.add_subplot()
        shap.plots.bar(shap_obj, show=False, max_display=6)
        plt.savefig('./figs/shap_bar_'+fe+'.png', dpi=500, bbox_inches='tight')
        f2 = plt.figure()
        ax2 = f2.add_subplot()
        shap.plots.beeswarm(shap_obj, show=False, max_display=6)    
        plt.savefig('./figs/shap_swarm_'+fe+'.png', dpi=500, bbox_inches='tight')
    #=================================================================
    # GPR Training by Random sampling
    X_sample_r = X_sample_in.copy()
    y_sample_r = y_sample_in.copy()
    
    # Initialize the training
    X_train = []
    y_train = []
    k = 0
    for k in range (len(X_ini)):
        X_train.append(X_ini[k])
        y_train.append(y_ini[k])
    metrics_rand = []
    # Training loop
    for j in range (n_samples+1):
        new_idx_r = rand(len(X_sample_r))[0]
        X_train.append(X_sample_r[new_idx_r])
        y_train.append(y_sample_r[new_idx_r])
        gpr.fit(np.array(X_train), np.array(y_train))
        # Delete the query from the samples space to avoid reselection 
        X_sample_r = np.delete(X_sample_r, new_idx_r, axis=0)
        y_sample_r = np.delete(y_sample_r, new_idx_r, axis=0)
        # Calculate metrics every five samples
        if (j == n_metrics[k]):
            y_pred = gpr.predict(X_test_scale)
            y_pred = y_pred.reshape(-1,1)
            y_pred_kj = scaler_y.inverse_transform(y_pred)
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
            r2 = r2_score(y_test, y_pred_kj)
            metrics_rand.append([j,rmse,r2])
            k = k+1
    return np.array(metrics), np.array(metrics_rand), np.array(pac_used)


# GPR ensemble for active learning 
def ENAL (df, n_samples, seed, pca, n_reg, fe):
    np.random.seed(seed)
    # Read data csv    
    df = pd.read_csv(df)
    # define X and y
    X = df.drop(['letters', 'disassoc', 'assoc'], axis=1)
    y = np.array(df[fe])
    if (pca):
        X.to_csv('./cases/PCA.csv', index=False)
        obj = DRAL(path_X='./cases/PCA.csv', seed=42)
        Xpca,X = obj.PCA(var=0.95) 
    # reshape y from vevtor to matrix
    y = y.reshape(-1,1)
    # Number of samples for validation
    X_sample, X_test, y_sample, y_test = train_test_split(X, 
                                                          y, 
                                                          test_size=0.32,
                                                          random_state=seed)
    # Scaling 
    # X
    scaler_x = MinMaxScaler()
    scaler_x.fit(X)
    X_sample_scale = scaler_x.transform(X_sample)
    X_test_scale = scaler_x.transform(X_test)
    # y
    scaler_y = MinMaxScaler()
    scaler_y.fit(y)
    y_sample_scale = scaler_y.transform(y_sample)
    
    # Start with n random samples
    idx_in=np.random.choice(len(X_sample_scale), size=n_reg, replace=False)
    X_ini = X_sample_scale[idx_in]
    y_ini = y_sample_scale[idx_in]
    X_sample = np.delete(X_sample_scale, idx_in, axis=0)
    y_sample = np.delete(y_sample_scale, idx_in, axis=0) 
    # Defining kernel for Gaussian Process Regressor 
    kernel = DotProduct()
    # Defining the active learner using modAL package 
    gpr = GaussianProcessRegressor(kernel=kernel,
                         random_state=0,
                         n_restarts_optimizer=0,
                         alpha=1)
    # Use GPR as an Active Learner, Max Std as a query strategy
    # initializing the regressors
   
    learner_list = [ActiveLearner(estimator=gpr,
                        X_training=X_ini[idx].reshape(1,-1), 
                        y_training=y_ini[idx].reshape(-1,1))
                for idx in range(len(idx_in))]

    # initializing the Committee
    committee = CommitteeRegressor(learner_list=learner_list,
                                   query_strategy=max_std_sampling)
    
    # To calculate metrics every 5 points
    n_metrics = np.arange(5,n_samples+5,5)
    # Empty lists to store the results
    metrics = []
    k = 0
    for i in range (n_samples+1):
        query_idx, query_instance = committee.query(X_sample)
        committee.teach(X_sample[query_idx].reshape(1,-1), 
                        y_sample[query_idx].reshape(-1,1))
        # Delete the query from the samples space to avoid reselection 
        X_sample = np.delete(X_sample, query_idx, axis=0)
        y_sample = np.delete(y_sample, query_idx, axis=0)
        # Metrics every 5 samples 
        if (i == n_metrics[k]):
            # Trained model Prediction on unseen data
            y_pred = committee.predict(X_test_scale)
            y_pred = y_pred.reshape(-1,1)
            y_pred_kj = scaler_y.inverse_transform(y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
            r2 = r2_score(y_test, y_pred_kj)
            metrics.append([i,rmse,r2])
            k=k+1
        joblib.dump(committee, "./trained_models/committee.pkl") 
    return np.array(metrics)





