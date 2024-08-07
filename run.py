from FEAL import FE_AL, plotting, ENAL
import numpy as np

# Original case
# Assoc
metrics, metrics_rd, pac_used = FE_AL(df='data_all.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=False,
                   pca=False,
                   fe='assoc',
                   use_shap=False)
                       
plotting(arr=metrics,
         arr_r=metrics_rd,
         name='original_assoc',
         unc=False,
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+0.5],
         y1_ticks=np.arange(2,13,2),
         y2_range=[-0.5,1.01],
         y2_ticks=np.arange(-1.0,1.01,0.50),
         legend_loc='lower right')

#Disassoc
metrics, metrics_rd, pac_used = FE_AL(df='data_all.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=False,
                   pca=False,
                   fe='disassoc',
                   use_shap=False)
                       
plotting(arr=metrics,
         arr_r=metrics_rd,
         name='original_disassoc',
         unc=False,
         #title1='(c)',
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+0.5],
         y1_ticks=np.arange(7,16,2),
         #title2='(d)',
         y2_range=[-0.5,1.01],
         y2_ticks=np.arange(-0.5,1.01,0.5),
         legend_loc='lower right')


#========================================================
# Drop A from sampling space
# From other monomers, change "noA".csv to "noC", "noE", ......
metrics, metrics_rd, pac_used = FE_AL(df='./cases/noA.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=True,
                   pca=False,
                   fe='assoc',
                   use_shap=False)
                       
#Disassoc
metrics, metrics_rd, pac_used = FE_AL(df='./cases/noA.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=True,
                   pca=False,
                   fe='disassoc',
                   use_shap=False)
                       
                     
#========================================================
# Apply PCA to reduce the number of features 
# Not included in the paper because I do not see a reason to reduce the feature space
# Assoc
metrics, metrics_rd, pac_used = FE_AL(df='data_all.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=False,
                   pca=True,
                   fe='assoc',
                   use_shap=False)
                       
#Disassoc
metrics, metrics_rd, pac_used = FE_AL(df='data_all.csv',
                   n_samples=100,
                   seed=42,
                   no_shuf=False,
                   pca=True,
                   fe='disassoc',
                   use_shap=False)
                       
#========================================================
# Testing model stability by running 100 case
# The data is shuffled at the beggining of each run 
# The run takes around 2.5 hours, the results are in ./cases/shuffle_assoc/diassoc.txt
fe = ['assoc', 'disassoc']
seed = np.random.choice(100,100,replace=False)
n_samples = np.arange(5,105,5)
for fe in fe:
    summary = []
    for j in range(len(n_samples)):
        lst = []
        lst_rand = []
        for i in range (0,len(seed)):
            metrics, metrics_rd, pac_used = FE_AL(df='data_all.csv',
                                        n_samples=n_samples[j],
                                        seed=seed[i],
                                        no_shuf=False,
                                        pca=False,
                                        fe=fe,
                                        use_shap=False)
            lst.append([i,metrics[-1,1],metrics[-1,2]])
            lst_rand.append([i,metrics_rd[-1,1],metrics_rd[-1,2]])
            a = np.array(lst)
            a_rand = np.array(lst_rand)
            print(fe)
            print("Number of samples =", n_samples[j])
            print ("i =",i,"/100")
            print(np.mean(a[:,1]),
                  np.mean(a[:,2]), 
                  np.mean(a_rand[:,1]), 
                  np.mean(a_rand[:,2]))
    
        summary.append([n_samples[j], 
                        np.mean(a[:,1]),
                        np.std(a[:,1]),
                        np.mean(a[:,2]),
                        np.std(a[:,2]),
                        np.mean(a_rand[:,1]),
                        np.std(a_rand[:,1]),
                        np.mean(a_rand[:,2]),
                        np.std(a_rand[:,2])])
        results = np.array(summary)
        np.savetxt('./cases/test_'+fe+'.txt', results)

# Plot the saved results
# Assoc 
data = np.loadtxt('./cases/results_assoc.txt')
metrics = data[:,(0,1,3,2,4)]
metrics_rd = data[:, (0,5,7,6,8)]

plotting(arr=metrics,
         arr_r=metrics_rd,
         name='unc_assoc',
         unc=True,
         #title1='(c)',
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+1.5],
         y1_ticks=np.arange(2,12,2),
         #title2='(d)',
         y2_range=[-0.5,1.01],
         y2_ticks=np.arange(-0.5,1.01,0.5),
         legend_loc='lower right')

# Disassoc
data = np.loadtxt('./cases/results_disassoc.txt')
metrics = data[:,(0,1,3,2,4)]
metrics_rd = data[:, (0,5,7,6,8)]

plotting(arr=metrics,
         arr_r=metrics_rd,
         name='unc_disassoc',
         unc=True,
         y1_range=[min(metrics[:,1])-0.5,max(metrics[:,1])+1.5],
         y1_ticks=np.arange(7,15,2),
         #title2='(d)',
         y2_range=[0.25,0.90],
         y2_ticks=np.arange(0.25,0.90,0.15),
         legend_loc='lower right')


#===============================================================
# Using more than one GPR 

R2 = []
metrics = ENAL(df='data_all.csv', 
               n_samples=100, 
               seed=42, 
               pca=False, 
               n_reg=10,
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