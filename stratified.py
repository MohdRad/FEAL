import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from FEAL import GP_regression_std, rand
from sklearn.gaussian_process.kernels import DotProduct
from modAL.models.learners import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class stratified ():
    def __init__(self, data_path:str, seed:int, fe:str, n_samples:int, n_steps:int, alpha:float):
        self.data_path = data_path
        self.seed = seed
        self.fe = fe
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.alpha = alpha
      
    def data_prep (self, T):
        data = pd.read_csv(self.data_path)
        X = data.drop(['letters','assoc','disassoc'], axis=1)
        y = np.array(data[self.fe]).reshape(-1,1)
        train, test = train_test_split(data, test_size=0.32, random_state=self.seed)
        Temp = train[train['T']==T]
        X_sample = Temp.drop(['letters','assoc','disassoc'], axis=1)
        y_sample = Temp[self.fe]
        X_test = test.drop(['letters','assoc','disassoc'], axis=1)
        y_test = test[self.fe]
    
        # reshape y from vevtor to matrix
        y_sample = np.array(y_sample).reshape(-1,1)
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
        return X_sample_scale, y_sample_scale, X_test_scale, y_test, y

    def initial_AL (self,T):
        np.random.seed(self.seed)
        X_sample, y_sample, X_test, y_test, y = self.data_prep(T)   
        # Start with one random sample
        idx_in = np.random.choice(len(X_sample), size=1, replace=False)
        X_ini = X_sample[idx_in]
        y_ini = y_sample[idx_in]
        
        X_sample = np.delete(X_sample, idx_in, axis=0)
        y_sample = np.delete(y_sample, idx_in, axis=0) 
        X_sample_in = X_sample.copy()
        y_sample_in = y_sample.copy()
        return X_sample, y_sample, X_ini, y_ini, X_sample_in, y_sample_in
    
    def AL_strata (self):
        np.random.seed(self.seed)
        # Read data csv    
        df = pd.read_csv(self.data_path)
        # define X and y
        X = df.drop(['letters','assoc','disassoc'], axis=1)
        y = df[self.fe]
        X_sample, X_test, y_sample, y_test = train_test_split(X, y,  
                                                                 test_size=0.32,
                                                                 random_state=self.seed)
        
       
        # reshape y from vevtor to matrix
        y_sample = np.array(y_sample).reshape(-1,1)
        y_test = np.array(y_test).reshape(-1,1)
        # Scaling 
        # X
        y = np.array(df[self.fe]).reshape(-1,1)
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
                            alpha=self.alpha)
                         
        # Start with one random sample
        idx_in=np.random.choice(len(X_sample_scale), size=1, replace=False)
        X_ini = X_sample_scale[idx_in]
        y_ini = y_sample_scale[idx_in]
        X_sample = np.delete(X_sample_scale, idx_in, axis=0)
        y_sample = np.delete(y_sample_scale, idx_in, axis=0) 

                        
        # Use GPR as an Active Learner, Max Std as a query strategy
        regressor = ActiveLearner(estimator=gpr,
                                 query_strategy=GP_regression_std,
                                 X_training=X_ini,
                                 y_training=y_ini)
        
        # To calculate metrics every 5 points
        n_metrics = np.arange(3,3*self.n_steps+3,3)
        # Empty lists to store the results
        metrics = []
        X_train = []
        k = 0
        for i in range (self.n_samples):
            query_idx, query_instance = regressor.query(X_sample)
            regressor.teach(X_sample[query_idx].reshape(1,-1), 
                            y_sample[query_idx].reshape(-1,1))
            X_train.append(query_instance)
            # Delete the query from the samples space to avoid reselection 
            X_sample = np.delete(X_sample, query_idx, axis=0)
            y_sample = np.delete(y_sample, query_idx, axis=0)
            # Metrics every 5 samples 
            if (i == n_metrics[k]-1):
                # Trained model Prediction on unseen data
                y_pred = regressor.predict(X_test_scale)
                y_pred = y_pred.reshape(-1,1)
                global y_pred_kj
                y_pred_kj = scaler_y.inverse_transform(y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
                r2 = r2_score(y_test, y_pred_kj)
                metrics.append([n_metrics[k],rmse,r2])
                k=k+1
        # save the model 
        joblib.dump(regressor, "./trained_models/"+self.fe+'_strata_'+str(self.n_steps*3+1)+".pkl")    
        return np.array(metrics)
    
    def rand_strata (self):
        np.random.seed(self.seed)
        X_sample_750, y_sample_750, X_test, y_test, y = self.data_prep(750) 
        X_sample_1000, y_sample_1000, X_ini, y_ini, X_sample_in_1000, y_sample_in_1000 = self.initial_AL(1000)
        X_sample_1250, y_sample_1250, X_test, y_test, y = self.data_prep(1250)
        X_sample_1 = X_sample_750.copy()
        y_sample_1 = y_sample_750.copy()
        X_sample_2 = X_sample_in_1000.copy()
        y_sample_2 = y_sample_in_1000.copy()
        X_sample_3 = X_sample_1250.copy()
        y_sample_3 = y_sample_1250.copy()
        #global X_sample_r
        #X_sample_r = np.concatenate((X_sample_1, X_sample_2, X_sample_3), axis=0)
        #y_sample_r = np.concatenate((y_sample_1, y_sample_2, y_sample_3), axis=0)
        
        scaler_y = MinMaxScaler()
        scaler_y.fit(y)
        
        kernel = DotProduct()
        gpr = GaussianProcessRegressor(kernel=kernel,
                            random_state=0,
                            n_restarts_optimizer=0,
                            alpha=self.alpha)
        
        # Initialize the training
        X_train = []
        y_train = []
        for k in range (len(X_ini)):
            X_train.append(X_ini[k])
            y_train.append(y_ini[k])
        metrics_rand = []
        # Training loop
        for j in range (self.n_steps):
            new_idx_r1 = rand(len(X_sample_1))[0]
            X_train.append(X_sample_1[new_idx_r1])
            y_train.append(y_sample_1[new_idx_r1])
            
            new_idx_r2 = rand(len(X_sample_2))[0]
            X_train.append(X_sample_2[new_idx_r2])
            y_train.append(y_sample_2[new_idx_r2])
            
            new_idx_r3 = rand(len(X_sample_3))[0]
            X_train.append(X_sample_3[new_idx_r3])
            y_train.append(y_sample_3[new_idx_r3])
            
            gpr.fit(np.array(X_train), np.array(y_train))
            # Delete the query from the samples space to avoid reselection 
            X_sample_1 = np.delete(X_sample_1, new_idx_r1, axis=0)
            y_sample_1 = np.delete(y_sample_1, new_idx_r1, axis=0)
            
            X_sample_2 = np.delete(X_sample_2, new_idx_r2, axis=0)
            y_sample_2 = np.delete(y_sample_2, new_idx_r2, axis=0)
            
            X_sample_3 = np.delete(X_sample_3, new_idx_r3, axis=0)
            y_sample_3 = np.delete(y_sample_3, new_idx_r3, axis=0)
            # Calculate metrics every three samples
            y_pred = gpr.predict(X_test)
            y_pred = y_pred.reshape(-1,1)
            y_pred_kj = scaler_y.inverse_transform(y_pred)
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_kj))
            r2 = r2_score(y_test, y_pred_kj)
            metrics_rand.append([3*(j+1),rmse,r2])

        return np.array(metrics_rand)

# =========================================================================    
# Execute the class 
def run_strata (data_path, fe, n_samples, n_steps, alpha):
    seed = np.arange(0,100,1)
    n_samples = np.arange(3,n_samples+3,3)
    n_steps = np.arange(1,n_steps+1,1)
    global metrics_AL, metrics_rd
    metrics_AL = np.zeros((len(n_steps), 9))
    metrics_rd = np.zeros((len(n_steps), 5))

    for j in range (len(n_steps)):
        met_al = []
        met_rand = []
        for i in range(len(seed)):
            strata = stratified(data_path = data_path, 
                                seed = seed[i], 
                                fe = fe,
                                n_samples = n_samples[j],
                                n_steps = n_steps[j],
                                alpha= alpha)

            metrics = strata.AL_strata()
            met_al.append(metrics[-1, (1,2)])
        
            metrics_rand = strata.rand_strata()
            met_rand.append(metrics_rand[-1, (1,2)])
            if ((i+1)%10 == 0):
                print ('N_samples', 3*n_steps[j], 'split', (i+1), 'complete')
    
        met_al_arr = np.array(met_al)
        met_rand_arr = np.array(met_rand)
        rmse_diff = met_rand_arr[:,0] - met_al_arr[:,0]
        r2_diff = met_al_arr[:,1] - met_rand_arr[:,1]

        al_mean = np.mean(met_al_arr, axis=0)
        rand_mean = np.mean(met_rand_arr, axis=0)
        al_std = np.std(met_al_arr, axis=0)
        rand_std = np.std(met_rand_arr, axis=0)
        # diff
        rmse_diff_mean = np.mean(rmse_diff)
        rmse_diff_std = np.std(rmse_diff)
        r2_diff_mean = np.mean(r2_diff)
        r2_diff_std = np.std(r2_diff)
        # indexing
        # AL
        metrics_AL[j,0] = 3*n_steps[j]
        metrics_AL[j,1] = al_mean[0]
        metrics_AL[j,2] = al_std[0]
        metrics_AL[j,3] = al_mean[1]
        metrics_AL[j,4] = al_std[1]
        metrics_AL[j,5] = rmse_diff_mean
        metrics_AL[j,6] = rmse_diff_std
        metrics_AL[j,7] = r2_diff_mean
        metrics_AL[j,8] = r2_diff_std
        # Random
        metrics_rd[j,0] = 3*n_steps[j]
        metrics_rd[j,1] = rand_mean[0]
        metrics_rd[j,2] = rand_std[0]
        metrics_rd[j,3] = rand_mean[1]
        metrics_rd[j,4] = rand_std[1]
    
    p = Path('cases')
    p.mkdir(parents=True, exist_ok=True)
    df_al = pd.DataFrame(metrics_AL, columns=['n_samples', 'RMSE mean', 'RMSE std',
                                                  'R2 mean', 'R2 std', 'RMSE diff mean',
                                                  'RMSE diff std', 'R2 diff mean', 'R2 diff std'])
    df_al.to_csv('./cases/strat_AL_'+fe+'.csv', index=False)
    df_rand = pd.DataFrame(metrics_rd, columns=['n_samples', 'RMSE mean', 'RMSE std', 'R2 mean', 'R2 std'])
    df_rand.to_csv('./cases/strat_rand_'+fe+'.csv', index=False)

# alpha tuning 
def tuning_strata (alpha, fe, seed):
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
            strata = stratified(data_path = 'data_all.csv', 
                                seed = seed[i], 
                                fe = fe,
                                n_steps = 33,
                                n_samples = 100,
                                alpha= alpha[j])

            metrics = strata.AL_strata()
            met_al.append(metrics[-1, (1,2)])
            metrics_rd = strata.rand_strata()
            met_rand.append(metrics_rand[-1, (1,2)])
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
    df_al.to_csv('./cases/tuning/alpha_strata_AL_'+fe+'.csv', index=False)
    df_rand = pd.DataFrame(metrics_rand, columns=['n_samples', 'RMSE kJ/mol', 'R2'])
    df_rand.to_csv('./cases/tuning/alpha_rand_'+fe+'.csv', index=False)   
