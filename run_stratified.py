from stratified import run_strata
from FEAL import plotting
import pandas as pd
import numpy as np

# Run and save the results
# Association FE

run_strata(data_path='data_all.csv',
           fe='assoc', 
           n_samples=99,
           n_steps=33, 
           alpha=1)

# Disassoc FE

run_strata(data_path='data_all.csv',
           fe='disassoc', 
           n_samples=99,
           n_steps=33, 
           alpha = 1)





# Plot the saved results
# Association
data_al = pd.read_csv('./cases/strat_AL_assoc.csv')
metrics = np.array(data_al[['n_samples', 'RMSE mean', 'R2 mean', 'RMSE std', 'R2 std']])
data_rd = pd.read_csv('./cases/strat_rand_assoc.csv')
metrics_rd = np.array(data_rd[['n_samples', 'RMSE mean', 'R2 mean', 'RMSE std', 'R2 std']])
diff = np.array(data_al[['n_samples', 'RMSE diff mean', 'R2 diff mean', 'RMSE diff std', 'R2 diff std']])

plotting(arr=metrics,
         arr_r=metrics_rd,
         stratified=True,
         diff = diff,
         smoothing=False,
         name='strat_assoc',
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+1.5],
         y1_ticks=np.arange(2,13,2),
         y2_range=[-0.9,1.01],
         y2_ticks=np.arange(-0.9,1.01,0.45),
         y3_range= [-2.6, 2.3],
         y3_ticks=[-2, -1, 0, 1.0, 2.0],
         y4_range=[-0.8,0.45],
         y4_ticks=[-0.8, -0.4, 0, 0.4],
         legend_loc='lower right')

# Diassoc
data_al = pd.read_csv('./cases/strat_AL_disassoc.csv')
metrics = np.array(data_al[['n_samples', 'RMSE mean', 'R2 mean', 'RMSE std', 'R2 std']])
data_rd = pd.read_csv('./cases/strat_rand_disassoc.csv')
metrics_rd = np.array(data_rd[['n_samples', 'RMSE mean', 'R2 mean', 'RMSE std', 'R2 std']])
diff = np.array(data_al[['n_samples', 'RMSE diff mean', 'R2 diff mean', 'RMSE diff std', 'R2 diff std']])

plotting(arr=metrics,
         arr_r=metrics_rd,
         stratified=True,
         diff = diff,
         smoothing=False,
         name='strat_disassoc',
         y1_range=[min(metrics[:,1])-0.7,max(metrics[:,1])+2.5],
         y1_ticks=np.arange(6,19,4),
         y2_range=[0.05,0.95],
         y2_ticks=np.arange(0.05,0.96,0.15),
         y3_range= [-3.8, 3.0],
         y3_ticks=[-3.0, 0, 3],
         y4_range=[-0.35, 0.25],
         y4_ticks=[-0.25, 0, 0.25],
         legend_loc='lower right')
