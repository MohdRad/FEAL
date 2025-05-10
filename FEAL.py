import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct,
                                              RBF,
                                              Matern,
                                              RationalQuadratic,
                                              ExpSineSquared)
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
from pathlib import Path
from sklearn import manifold
from sklearn.decomposition import PCA
#import umap

# For active learning, we shall define a custom query strategy 
# tailored to Gaussian processes (gp).
# Select the prediction y_gp that has the largest standard deviation 
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

#==============================================================================
# Function for Random sample selection 
def rand(domain):
    new_idx = np.random.choice(domain, size=1, replace=False)
    return new_idx

#==============================================================================
# Function to make curves smoother
def smoother (x,y):
    spl = make_interp_spline(x,y,k=3)
    x_new = np.linspace(min(x),max(x),300)
    y_smooth = spl(x_new)
    return x_new, y_smooth

  
#==============================================================================
# Function to plot the output of FE_AL function introduced down
def plotting (arr, 
              arr_r,
              stratified,
              diff,
              smoothing,
              name, 
              y1_range,
              y1_ticks,
              y2_range,
              y2_ticks,
              y3_range,
              y3_ticks,
              y4_range,
              y4_ticks,
              legend_loc):
    '''
    Parameters
    ----------
    arr:        (arr) metrics of AL at n_samples
    arr_r:      (arr) metrics of random at n_samples
    name:       (str) name of output plot file 
    unc:        (bool) choices are True or False, to include uncertainty
    y1_range:   (list) [min,max] of RMSE  
    y1_ticks:   (list/array) RMSE ticks elements, use numpy.arange 
    y2_range:   (list) [min,max] of R2  
    y2_ticks:   (list/array) R2 ticks elements, use numpy.arange
    y3_range:   (list/array)
    Returns
    -------
    saves plot in ./figs

    '''
    # RMSE
    plt.rcParams.update({'font.size': 18})
    if (smoothing):
        n_new, rmse_smooth = smoother(arr[:,0], arr[:,1])
        n_new, r2_smooth = smoother(arr[:,0], arr[:,2])
        n_new, rmse_r_smooth = smoother(arr_r[:,0], arr_r[:,1])
        n_new, r2_r_smooth = smoother(arr_r[:,0], arr_r[:,2])
        n_new, rmse_unc_smooth = smoother(arr[:,0],arr[:,3])
        n_new, rmse_r_unc_smooth = smoother(arr_r[:,0],arr_r[:,3])
        n_new, r2_unc_smooth = smoother(arr[:,0],arr[:,4])
        n_new, r2_r_unc_smooth = smoother(arr_r[:,0],arr_r[:,4])
        n_new, diff_rmse_smooth = smoother(diff[:,0], diff[:,1])
        n_new, diff_rmse_unc_smooth = smoother(diff[:,0], diff[:,3])
        n_new, diff_r2_smooth = smoother(diff[:,0], diff[:,2])
        n_new, diff_r2_unc_smooth = smoother(diff[:,0], diff[:,4])
    else: 
        n_new, rmse_smooth = arr[:,0], arr[:,1]
        n_new, r2_smooth = arr[:,0], arr[:,2]
        n_new, rmse_r_smooth = arr_r[:,0], arr_r[:,1]
        n_new, r2_r_smooth = arr_r[:,0], arr_r[:,2]
        n_new, rmse_unc_smooth = arr[:,0], arr[:,3]
        n_new, rmse_r_unc_smooth = arr_r[:,0], arr_r[:,3]
        n_new, r2_unc_smooth = arr[:,0], arr[:,4]
        n_new, r2_r_unc_smooth = arr_r[:,0], arr_r[:,4]
        n_new, diff_rmse_smooth = diff[:,0], diff[:,1]
        n_new, diff_rmse_unc_smooth = diff[:,0], diff[:,3]
        n_new, diff_r2_unc_smooth = diff[:,0], diff[:,4]
        n_new, diff_r2_smooth = diff[:,0], diff[:,2]
    
    fig,ax1 = plt.subplots()
    ax1.set_xlabel('Number of queries')
    ax1.set_ylabel('RMSE kJ/mol')
    ax1.set_xlim([0,max(arr[:,0])+1.0])
    ax1.set_ylim(y1_range)
    ax1.set_yticks(y1_ticks)
    ax1.plot(n_new, rmse_smooth, color='tab:red',label='AL')
    if (stratified):
        ax1.plot(n_new,rmse_r_smooth, '--', color='black', label='Stratified')
    else:
        ax1.plot(n_new,rmse_r_smooth, '--', color='black', label='Random')
    plt.legend(loc='best')
    plt.fill_between(n_new, rmse_smooth-rmse_unc_smooth, rmse_smooth+rmse_unc_smooth,
                         color='red', alpha=0.3)
    plt.fill_between(n_new, rmse_r_smooth-rmse_r_unc_smooth, rmse_r_smooth+rmse_r_unc_smooth,
                         color='grey', alpha=0.5)
    plt.title("(a)", y=-0.35)
    plt.legend(loc='best')
    p = Path('./figs/'+name)
    p.mkdir(parents=True, exist_ok=True)
    plt.savefig('./figs/'+name+'/'+name+'_rmse.png', dpi=500, bbox_inches='tight')
    
    # R2
    fig,ax2 = plt.subplots()
    ax2.set_xlabel('Number of queries')
    ax2.set_ylabel('R$^2$')
    ax2.set_xlim([0,max(arr[:,0])+1.0])
    ax2.set_ylim(y2_range)
    ax2.set_yticks(y2_ticks)
    ax2.plot(n_new, r2_smooth, color='tab:blue',label='AL')
    if (stratified):
        ax2.plot(n_new, r2_r_smooth, '--', color='black', label='Stratified')
    else:
        ax2.plot(n_new, r2_r_smooth, '--', color='black', label='Random')
    ax2.yaxis.label.set(rotation='horizontal', ha='right')
    plt.title("(b)", y=-0.35)
    plt.legend(loc=legend_loc)

    plt.fill_between(n_new, r2_smooth-r2_unc_smooth, r2_smooth+r2_unc_smooth,
                         color='blue', alpha=0.3)
    plt.fill_between(n_new, r2_r_smooth-r2_r_unc_smooth, r2_r_smooth+r2_r_unc_smooth,
                         color='grey', alpha=0.5)
    plt.savefig('./figs/'+name+'/'+name+'_r2.png', dpi=500, bbox_inches='tight')
    
    # RMSE diff
    fig, ax3 = plt.subplots()
    ax3.set_xlabel('Number of queries')
    ax3.set_ylabel('RMSE$_{Rand}$ - RMSE$_{AL}$')
    ax3.set_xlim([0,max(arr[:,0])+1.0])
    ax3.set_ylim(y3_range)
    ax3.set_yticks(y3_ticks)
    ax3.plot(n_new, diff_rmse_smooth, color='tab:red')    
    ax3.plot([0,1000], [0,0], color='black', linestyle='dashed')  
    plt.fill_between(n_new, diff_rmse_smooth-diff_rmse_unc_smooth, diff_rmse_smooth+diff_rmse_unc_smooth,
                         color='red', alpha=0.3)
    plt.title("(c)", y=-0.35)
    plt.savefig('./figs/'+name+'/'+name+'_rmse_diff.png', dpi=500, bbox_inches='tight')
    
    # R2 diff
    fig, ax4 = plt.subplots()
    ax4.set_xlabel('Number of queries')
    ax4.set_ylabel('R$^2_{AL}$ - R$^2_{Rand}$')
    ax4.set_xlim([0,max(arr[:,0])+1.0])
    ax4.set_ylim(y4_range)
    ax4.set_yticks(y4_ticks)
    ax4.plot(n_new, diff_r2_smooth, color='tab:blue')
    ax4.plot([0,1000], [0,0], color='black', linestyle='dashed')  
    #ax4.plot(n_new, np.zeros(300), color='black', linestyle='dashed')    
    plt.fill_between(n_new, diff_r2_smooth-diff_r2_unc_smooth, diff_r2_smooth+diff_r2_unc_smooth,
                         color='blue', alpha=0.3)
    plt.title("(d)", y=-0.35)
    plt.savefig('./figs/'+name+'/'+name+'_r2_diff.png', dpi=500, bbox_inches='tight')
    
    

#=============================================================================
def tuning_visual (data_path, name, title, xlim, ylim, xticks, yticks):
    plt.rcParams.update({'font.size': 18})
    data = pd.read_csv(data_path)
    alpha = data['alpha']
    R2 = data['R2']
    plt.figure()
    plt.scatter(alpha, R2, s=75)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel('R$^2_{AL}$')
    plt.xlabel('$\\alpha$')
    plt.title(title, y=-0.35)
    p = Path('figs/tuning')
    p.mkdir(parents=True, exist_ok=True)
    plt.savefig('./figs/tuning/'+name+'.png', dpi=500, bbox_inches='tight')
    
def error_hist (data_path, name, xlim, ylim, xticks, yticks, title):
    global data
    data = pd.read_csv(data_path)
    error = data['RPE']
    p = Path('./figs/histogram')
    p.mkdir(parents=True, exist_ok=True)
    hist, bin_edges = np.histogram(error, bins='auto')
    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.hist(error, bins=len(hist), color = 'tab:red', edgecolor='black')
    plt.ylabel('Frequency')
    plt.xlabel('Relative Percent Error')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title(title, y=-0.35)
    plt.savefig('./figs/histogram/'+name+'_error_distribution.png', dpi=500, bbox_inches='tight')
    
    
#==============================================================================
# Holdout 
def hold_out_set (path, letter, fe):
    df = pd.read_csv(path)
    cols = df.columns
    df_arr = np.array(df)
    sample_list = []
    test_list = []
    for i in range (len(df_arr)):
        if (letter in df_arr[i,310]):
            test_list.append(df_arr[i])
        else:
            sample_list.append(df_arr[i])
    samples = np.array(sample_list)
    samples = pd.DataFrame(samples,columns=cols)
    tests = np.array(test_list)
    tests = pd.DataFrame(tests,columns=cols)
    samples = samples.sample(len(df_arr) - round(0.32*len(df_arr)), random_state=42)
    X_sample = samples.drop(['letters','assoc','disassoc'], axis=1)
    y_sample = samples[fe]
    X_test = tests.drop(['letters','assoc','disassoc'], axis=1)
    y_test = tests[fe]
    return X_sample, X_test, y_sample, y_test

#==============================================================================
# GPR training using Active Learning    
def FE_AL (path, n_samples, seed, pca, fe, use_shap, hold_out, letter, ref_case, alpha):
    '''
    Parameters
    ----------
    df :        (str)  The name and directory of the data frame (.csv)
    n_samples : (int)  Number of samples used in the active learning
    seed:       (int)  The way the data will be shuffled
    pca:        (bool) choices are 'True' OR 'False', apply PCA 
    fe:         (str)  free energy to be used as an output, choices are 'assoc', 'disassoc'
    use_shap:   (bool) Use SHAP for explainable AI
    hold_out:   (bool) Test on a molecule not in the sampling space 
    letter:     (str) Choices are {A, C, D, E, G, J, P, R}, set to None if holdout is False
    
    Returns
    -------
    metrics: (array) an array with number of queries, Root Mean Squared Error, 
                     and cofficient of determination 

    '''
    np.random.seed(seed)
    # Read data csv    
    df = pd.read_csv(path)
    # define X and y
    X = df.drop(['letters','assoc','disassoc'], axis=1)
    y = df[fe]
    if (pca):
        X.to_csv('./cases/PCA.csv', index=False)
        obj = DRAL(path_X='./cases/PCA.csv')
        X = obj.PCA(var=0.95)
    
    if (hold_out):
        X_sample, X_test, y_sample, y_test = hold_out_set(path, letter, fe)
        

    else: 
        X_sample, X_test, y_sample, y_test = train_test_split(X, y,  
                                                             test_size=0.32,
                                                             random_state=seed)
    
    if (ref_case):
        X_test = X_test[:42]
        y_test = y_test[:42]
    
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
    #kernel = RBF(length_scale=0.1)
    #kernel = Matern(length_scale=1, length_scale_bounds=(1e-5,1e6))
    #kernel = RationalQuadratic(length_scale_bounds=(1e-5,1e6))
    #kernel = ExpSineSquared(length_scale = 0.01, length_scale_bounds=(1e-5,1e8))
    # Defining the active learner using modAL package 
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                        random_state=0,
                        n_restarts_optimizer=0,
                        alpha=alpha)
                     
    # Start with one random sample
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
    n_metrics = np.arange(5,n_samples+5,5)
    # Empty lists to store the results
    metrics = []
    X_train = []
    k = 0
    for i in range (n_samples+1):
        query_idx, query_instance = regressor.query(X_sample)
        regressor.teach(X_sample[query_idx].reshape(1,-1), 
                        y_sample[query_idx].reshape(-1,1))
        X_train.append(query_instance)
        # Delete the query from the samples space to avoid reselection 
        X_sample = np.delete(X_sample, query_idx, axis=0)
        y_sample = np.delete(y_sample, query_idx, axis=0)
        # Metrics every 5 samples 
        if (i == n_metrics[k]):
            # Trained model Prediction on unseen data
            y_pred = regressor.predict(X_test_scale)
            y_pred = y_pred.reshape(-1,1)
            global y_pred_kj
            y_pred_kj = scaler_y.inverse_transform(y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
            r2 = r2_score(y_test, y_pred_kj)
            metrics.append([i,rmse,r2])
            k=k+1
    # save the model 
    joblib.dump(regressor, "./trained_models/"+fe+str(n_samples)+".pkl")
    
    if(use_shap):
        p = Path('cases/hist')
        p.mkdir(parents=True, exist_ok=True)
        train, test = train_test_split(df,  
                                       test_size=0.32,
                                       random_state=seed)
        pacs = np.array(test[['letters', 'T', fe]])
        rpe = (100*np.absolute(pacs[:,2]-y_pred_kj[:,0]))/pacs[:,2]
        error = np.concatenate((pacs, y_pred_kj, rpe.reshape(-1,1)),  axis=1)
        error_df = pd.DataFrame(error, columns=['PAC', 'T', 'True', 'Pred', 'RPE'])
        error_df.to_csv('./cases/hist/hist_'+fe+'.csv', index=False)
        
        X_train_summary = shap.kmeans(np.array(X_train), 10)
        explainer = shap.KernelExplainer(model = regressor.predict, data = X_train_summary)
        X_test_shap = pd.DataFrame(X_test_scale, columns=X.columns)
        shap_obj = explainer(X_test_shap)
        shap_values = explainer.shap_values(X_test_shap)
        avg_shap = np.mean(np.abs(shap_values), axis=0)
        cols = np.array(X.columns)
        indices = avg_shap.argsort()
        top10 = indices[len(indices)-10:]
        p = Path('./figs/shap')
        p.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.size'] = 18
        f1 = plt.figure()
        ax1 = f1.add_subplot()
        #shap.plots.bar(shap_obj, show=False, max_display=10)
        f1 = plt.figure()
        plt.barh(range(10), avg_shap[top10], color='dodgerblue', align='center')
        plt.yticks(range(10), cols[top10])
        if (fe=='assoc'):
            plt.xticks([0,0.05,0.10,0.15,0.2, 0.25,0.30, 0.35])
        elif(fe=='disassoc'):
            plt.xticks([0,0.05,0.10])
        plt.xlabel('mean(|SHAP Values|)')
        for index, value in enumerate(avg_shap[top10]):
            plt.text(value, index-0.25, str("{:.4f}".format(value)))
        plt.title('(a)', y=-0.35)
        plt.savefig('./figs/shap/shap_bar_'+fe+'.png', dpi=500, bbox_inches='tight')
        
        f2 = plt.figure()
        ax2 = f2.add_subplot()
        shap.summary_plot(shap_obj, show=False, max_display=10)
        fig, ax = plt.gcf(), plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=18)
        if (fe=='assoc'):
            ax.set_xticks([-0.3, 0, 0.3])
        elif(fe=='disassoc'):
            ax.set_xticks([-0.1, 0, 0.1])
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=18)
        plt.title('(b)', y=-0.25)
        plt.savefig('./figs/shap/shap_swarm_'+fe+'.png', dpi=500, bbox_inches='tight')
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
    return np.array(metrics), np.array(metrics_rand)


#=============================================================================
def tuning (alpha, fe, seed):
    np.random.seed(seed)
    seed = np.random.randint(0, 40, 9)
    seed = seed.tolist() 
    seed.append(42)
    metrics_AL = np.zeros((len(alpha), 3))
    metrics_rand = np.zeros((len(alpha), 3))
    for j in range (len(alpha)):
        met_al = []
        met_rand = []
        for i in range(len(seed)):
            metrics, metrics_rd = FE_AL(path='data_all.csv',
                                        n_samples=100,
                                        seed=seed[i],
                                        pca=False,
                                        fe=fe,
                                        use_shap=False,
                                        hold_out=False,
                                        letter=None,
                                        ref_case=False,
                                        alpha=alpha[j])
            met_al.append(metrics[-1, (1,2)])
            met_rand.append(metrics_rd[-1, (1,2)])
        met_al_arr = np.array(met_al)
        met_rand_arr = np.array(met_rand)
        al_mean = np.mean(met_al_arr, axis=0)
        rand_mean = np.mean(met_rand_arr, axis=0)
        print (j,'/',len(alpha))
        print ('alpha =', alpha[j])
        print ('R2 =', al_mean[1])
        metrics_AL[j,0] = alpha[j]
        metrics_AL[j,1:] = al_mean
        metrics_rand[j,0] = alpha[j]
        metrics_rand[j,1:] = rand_mean
    p = Path('cases/tuning')
    p.mkdir(parents=True, exist_ok=True)
    df_al = pd.DataFrame(metrics_AL, columns=['alpha', 'RMSE kJ/mol', 'R2'])
    df_al.to_csv('./cases/tuning/alpha_AL_'+fe+'.csv', index=False)
    df_rand = pd.DataFrame(metrics_rand, columns=['n_samples', 'RMSE kJ/mol', 'R2'])
    df_rand.to_csv('./cases/tuning/alpha_rand_'+fe+'.csv', index=False)
    
    
#==============================================================================
# GPR ensemble for active learning 
def ENAL (df, n_samples, seed, pca, n_reg, fe):
    '''
    df:         (str)  The name and directory of the data frame (.csv)
    n_samples:  (int)  Number of samples used in the active learning
    seed:       (int)  The way the data will be shuffled
    pca:        (bool) choices are 'True' OR 'False', apply PCA 
    n_reg:      (int)  number of GP regressors in the ensemble  
    fe:         (str)  free energy to be used as an output, choices are 'assoc', 'disassoc'
    
    Returns
    -------
    Array of (n_samples, RMSE, R2)
    

    '''
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
    # Use GPR as an Active Learner, Max Std as a query strategy
    # initializing the regressors
   
    learner_list = [ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel,
                                                                     n_restarts_optimizer=0,
                                                                     alpha=1),
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
            y_pred = committee.predict(X_test_scale)
            y_pred = y_pred.reshape(-1,1)
            y_pred_kj = scaler_y.inverse_transform(y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
            r2 = r2_score(y_test, y_pred_kj)
            metrics.append([i,rmse,r2])
            k=k+1
        joblib.dump(committee, "./trained_models/committee.pkl") 
    return np.array(metrics)


#==============================================================================
# Function to get FE at 100 different sampling/testing splits
def fe_unc (fe):
    seed = np.arange(0,100,1)
    n_samples = np.arange(5,105,5)
    for k in range(len(fe)):
        summary = []
        for j in range(len(n_samples)):
            lst = []
            lst_rand = []
            for i in range (0,len(seed)):
                metrics, metrics_rd = FE_AL(path='data_all.csv',
                                            n_samples=n_samples[j],
                                            seed=seed[i],
                                            pca=False,
                                            fe=fe[k],
                                            use_shap=False,
                                            hold_out=False,
                                            letter=None,
                                            ref_case=False,
                                            alpha=1)

                lst.append([i, metrics[-1,1], metrics[-1,2]])
                lst_rand.append([i, metrics_rd[-1,1], metrics_rd[-1,2]])
                a = np.array(lst)
                a_rand = np.array(lst_rand)
                print(fe[k])
                print("Number of samples =", n_samples[j])
                print ("i =",i,"/100")
                print(np.mean(a[:,1]),
                      np.mean(a[:,2]), 
                      np.mean(a_rand[:,1]), 
                      np.mean(a_rand[:,2]))
            diff_rmse = a_rand[:,(1,2)]-a[:,(1,2)]
            diff_r2 = a[:,(1,2)]-a_rand[:,(1,2)]
            summary.append([n_samples[j],           # 0 Number of samples
                            np.mean(a[:,1]),        # 1 RMSE-AL-mean
                            np.std(a[:,1]),         # 2 RMSE-AL-std
                            np.mean(a[:,2]),        # 3 R2-AL-mean
                            np.std(a[:,2]),         # 4 R2-AL-Std
                            np.mean(a_rand[:,1]),   # 5 RMSE-Random-mean
                            np.std(a_rand[:,1]),    # 6 RMSE-Random-std
                            np.mean(a_rand[:,2]),   # 7 R2-Random-mean
                            np.std(a_rand[:,2]),    # 8 R2-Random-std
                            np.mean(diff_rmse[:,0]),# 9 RMSE: mean(Rand - AL)
                            np.mean(diff_r2[:,1]),  # 10 R2: mean(Rand - AL)
                            np.std(diff_rmse[:,0]), # 11 RMSE: Std(Rand-AL)
                            np.std(diff_r2[:,1])])  # 12 R2: Std(Rand-AL)
        results = np.array(summary)
        np.savetxt('./cases/test_'+fe[k]+'.txt', results)       

#==============================================================================
# Representer Theorem
def rep_theory (fe, xlim, title):
    df = pd.read_csv('./cases/shap.csv')
    X =  df.drop(['letters','assoc','disassoc'], axis=1)
    y = df[fe]
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.32,
                                                    random_state=42)
    gpr = GaussianProcessRegressor(kernel=DotProduct(sigma_0=1),
                               alpha=1,
                               random_state=0)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(np.array(y_train).reshape(-1,1))
    gpr.fit(X_train,y_train)
    alpha = gpr.alpha_.reshape(1,-1)
    w = alpha.dot(X_train)
    X_test = scaler_X.fit_transform(X_test)
    y_pred = gpr.predict(X_test)
    y_pred_2 = []
    for i in range(len(X_test)):
        y_pred_2.append(w.dot(X_test[i,:].reshape(-1,1))[0][0])
    
    p = Path('./figs/rep_theory')
    p.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(y_pred_2, y_pred)    
    plt.xlabel('$w\cdot x$')
    plt.ylabel('$gpr.predict()$')
    w_abs = np.absolute(w).reshape(-1,1)
    w_df = pd.DataFrame(w_abs, index=X.columns)
    w_df = w_df.sort_values(by=0,ascending=False)
    top10 = w_df[:10]
    top10 = top10.sort_values(by=0,ascending=True)
    plt.figure()
    top10.plot.barh(legend=False, color='dodgerblue')
    plt.xlabel ('|w|')
    plt.xlim(xlim)
    plt.title(title, y=-0.35)
    indices = np.array(top10.values).flatten()
    for index, value in enumerate(indices):
        plt.text(value, index-0.25, str("{:.4f}".format(value)))
    plt.savefig('./figs/rep_theory/rep_theory_'+fe+'.png', dpi=500, bbox_inches='tight')
#==============================================================================
# tSNE
def tSNE (data_path, red_type, drop_Temp):
    data = pd.read_csv(data_path)
    letters = np.array(data['letters'])
    if (drop_Temp):
        X = data.drop(['letters', 'assoc', 'disassoc', 'T'], axis=1)
    else:
        X = data.drop(['letters', 'assoc', 'disassoc'], axis=1)
    X_scaler = MinMaxScaler()
    Xs = X_scaler.fit_transform(X)
    
    if (red_type=='tSNE'):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(Xs)
    elif (red_type=='pca'):
        pca = PCA(n_components=2)
        X_tsne = pca.fit_transform(Xs)
    elif (red_type=='umap'):
        r_umap = umap.UMAP(random_state=42)
        X_tsne = r_umap.fit_transform(Xs)
    
    p = Path('./figs/tSNE')
    p.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.scatter(X_tsne[:,0], X_tsne[:,1])
    for i in range(len(X_tsne)):
        if ('A' in letters[i]):
            plt.scatter(X_tsne[i,0], X_tsne[i,1], color='orange', s=90)
    
        if ('C' in letters[i]):
            plt.scatter(X_tsne[i,0], X_tsne[i,1], color='red', s=75)
                
        if ('P' in letters[i]):
           plt.scatter(X_tsne[i,0], X_tsne[i,1], color='black', s=60)
    
        if ('K' in letters[i]):
            plt.scatter(X_tsne[i,0], X_tsne[i,1], color='green', s=40)
        if ('G' in letters[i]):
            plt.scatter(X_tsne[i,0], X_tsne[i,1], color='yellow', s=30)

    
    a = np.where(letters=='AA')[0][0]  
    plt.scatter(X_tsne[a,0], X_tsne[a,1], color='orange', s=75, label='A')        
    a = np.where(letters=='CC')[0][0]
    plt.scatter(X_tsne[a,0], X_tsne[a,1], color='red', s=60, label='C')
    a = np.where(letters=='PP')[0][0]      
    plt.scatter(X_tsne[a,0], X_tsne[a,1], color='black', s=40, label='I')
    a = np.where(letters=='KK')[0][0]
    plt.scatter(X_tsne[a,0], X_tsne[a,1], color='green', s=30, label='K')
    a = np.where(letters=='GG')[0][0]
    plt.scatter(X_tsne[a,0], X_tsne[a,1], s=30, color='tab:blue', label='Others')
    plt.legend(fontsize=13)
    if (red_type=='tSNE'):
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig('./figs/tSNE/tSNE_dissoc.png', dpi=500, bbox_inches='tight')
    elif (red_type=='pca'):
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.savefig('./figs/tSNE/pca_dissoc.png', dpi=500, bbox_inches='tight')
    elif (red_type=='umap'):
        plt.xlabel('U 1')
        plt.ylabel('U 2')
        plt.savefig('./figs/tSNE/umap_dissoc.png', dpi=500, bbox_inches='tight')
    
    

