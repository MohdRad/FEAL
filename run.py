from FEAL import FE_AL, plotting, ENAL, fe_unc, rep_theory
import numpy as np
             
#========================================================
# Testing model stability by running 100 case
# The data is shuffled at the beggining of each run 
# The run takes around 3 hours, the results are in ./cases/shuffle_assoc/diassoc.txt
# uncomment the following line if you want to run 
fe_unc(['assoc','disassoc'])

# Plot the saved results
# Association
data = np.loadtxt('./cases/test_assoc.txt')
metrics = data[:,(0,1,3,2,4)]
metrics_rd = data[:, (0,5,7,6,8)]
diff = data[:,(0,9,10,11,12)]

plotting(arr=metrics,
         arr_r=metrics_rd,
         diff = diff,
         name='unc_assoc',
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+1.5],
         y1_ticks=np.arange(2,12,2),
         y2_range=[-0.5,1.01],
         y2_ticks=np.arange(-0.5,1.01,0.5),
         y3_range= [-2, 2.5],
         y3_ticks=[-2, -1, 0, 1.0, 2.0],
         y4_range=[-0.5,0.5],
         y4_ticks=[-0.5, -0.25, 0, 0.25, 0.5],
         legend_loc='lower right')


# Disassoc
data = np.loadtxt('./cases/test_disassoc.txt')
metrics = data[:,(0,1,3,2,4)]
metrics_rd = data[:, (0,5,7,6,8)]
diff = data[:,(0,9,10,11,12)]

plotting(arr=metrics,
         arr_r=metrics_rd,
         name='unc_disassoc',
         diff=diff,
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+1.5],
         y1_ticks=np.arange(7,15,2),
         y2_range=[0.25,0.90],
         y2_ticks=np.arange(0.25,0.90,0.15),
         y3_range= [-2.5, 3.5],
         y3_ticks=[-2, -1, 0, 1.0, 2.0, 3],
         y4_range=[-0.25,0.28],
         y4_ticks=[-0.25, 0, 0.25],
         legend_loc='lower right')


#===============================================================
# Using more than one GPR 
n_reg = [1,3,5,7,9]
R2 = []
RMSE = []
for i in range(len(n_reg)):
    metrics = ENAL(df='data_all.csv', 
                   n_samples=100, 
                   seed=42, 
                   pca=False, 
                   n_reg=n_reg[i],
                   fe='disassoc')
    R2.append(metrics[-1,2])
    RMSE.append(metrics[-1,1])


#========================================================
# Dropout from sampling space
# In the paper: I==J, P==K, R==M
monomers = ['A', 'C', 'E', 'G', 'J', 'P', 'R']

# Association Free Energy
# Reference metrics to compare with holdout 
metrics, metrics_rd = FE_AL(path='data_all.csv',
                                          n_samples=100,
                                          seed=42,
                                          pca=False,
                                          fe='assoc',
                                          use_shap=False,
                                          hold_out=False,
                                          letter=None,
                                          ref_case=True)
print('assoc FE Ref. metrics using 40 samples =', metrics[7])
r2_ref = metrics[7,2]

# Holdout runs
for monomer in monomers:
    metrics, metrics_rd = FE_AL(path='data_all.csv',
                                          n_samples=100,
                                          seed=42,
                                          pca=False,
                                          fe='assoc',
                                          use_shap=False,
                                          hold_out=True,
                                          letter=monomer,
                                          ref_case=False)
    
    print (monomer,                                     # Molecule        
           round(metrics[7,1],2),                       # RMSE
           round(metrics[7,2],2),                       # R2
           round(100*(-metrics[7,2]+r2_ref)/(r2_ref),2)) #Delta R2

# Disassoc. Free Energy
# Reference Value 
metrics, metrics_rd = FE_AL(path='data_all.csv',
                                          n_samples=100,
                                          seed=42,
                                          pca=False,
                                          fe='disassoc',
                                          use_shap=False,
                                          hold_out=False,
                                          letter=None,
                                          ref_case=True)
r2_ref = metrics[13,2]
print('disassoc FE Ref. metrics using 70 samples =', metrics[13])
for monomer in monomers:
    metrics, metrics_rd = FE_AL(path='data_all.csv',
                                          n_samples=100,
                                          seed=42,
                                          pca=False,
                                          fe='disassoc',
                                          use_shap=False,
                                          hold_out=True,
                                          letter=monomer,
                                          ref_case=False)
    print (monomer, 
           round(metrics[13,1],2),
           round(metrics[13,2],2), 
           round(100*(-metrics[13,2]+r2_ref)/(r2_ref),2))

#===============================================================
# Use shap
metrics, metrics_rd = FE_AL(path='./cases/shap.csv',
                   n_samples=100,
                   seed=42,
                   pca=False,
                   fe='assoc',
                   use_shap=True,
                   hold_out=False,
                   letter=None,
                   ref_case=False)


metrics, metrics_rd  = FE_AL(path='./cases/shap.csv',
                   n_samples=100,
                   seed=42,
                   pca=False,
                   fe='disassoc',
                   use_shap=True,
                   hold_out=False,
                   letter=None,
                   ref_case=False)


#==============================================================================
# Use Representer Theorem
rep_theory ('assoc', [0,0.85])
rep_theory ('disassoc', [0,0.27])
