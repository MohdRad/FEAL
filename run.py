from FEAL import FE_AL, plotting, ENAL, fe_unc
import numpy as np
import matplotlib.pyplot as plt

#========================================================
# Drop A from sampling space
# From other monomers, change "noA".csv to "noC", "noE", ......
monomers = ['A', 'C', 'E', 'G', 'I', 'K', 'M']

for monomer in monomers:
    metrics, metrics_rd, pac_used = FE_AL(df='./cases/no'+monomer+'.csv',
                                          n_samples=100,
                                          seed=42,
                                          no_shuf=True,
                                          pca=False,
                                          fe='assoc',
                                          use_shap=False)
                       
    #Disassoc
    metrics, metrics_rd, pac_used = FE_AL(df='./cases/no'+monomer+'.csv',
                                          n_samples=100,
                                          seed=42,
                                          no_shuf=True,
                                          pca=False,
                                          fe='disassoc',
                                          use_shap=False)
                       
                     
#========================================================
# Testing model stability by running 100 case
# The data is shuffled at the beggining of each run 
# The run takes around 2.5 hours, the results are in ./cases/shuffle_assoc/diassoc.txt
# uncomment the two lines below if you want to run 
#fe_unc('assoc')
#fe_unc('disassoc')

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

R2 = []
metrics = ENAL(df='data_all.csv', 
               n_samples=100, 
               seed=42, 
               pca=False, 
               n_reg=5,
               fe='disassoc')
R2.append(metrics[-1,2])


#===============================================================
# Use shap
metrics, metrics_rd, pac_used = FE_AL(df='./cases/shap.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=False,
                   pca=False,
                   fe='assoc',
                   use_shap=True)


metrics, metrics_rd, pac_used = FE_AL(df='./cases/shap.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=False,
                   pca=False,
                   fe='disassoc',
                   use_shap=True)

