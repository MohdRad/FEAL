"""
Created on Wed May  1 18:41:57 2024
@author: mirad
"""

from FEAL import FE_AL, plotting, ENAL
import numpy as np

# Original case
metrics, metrics_rd = FE_AL(df = 'data_all.csv', 
                       n_samples = 30, 
                       spacing =2,
                       pca=False,
                       shuf=False) 
                       
plotting(metrics,metrics_rd,'./figs/original.png')

#=======================================================
# Drop A,B,C from the training data
metrics, metrics_rd = FE_AL(df = './cases/noABC.csv', 
                       n_samples = 30, 
                       spacing =2,
                       pca=False,
                       shuf=False) 
                       

plotting(metrics,metrics_rd,'./figs/noABC.png')

#========================================================
# Apply PCA to reduce the number of features 
metrics, metrics_rd = FE_AL(df = 'data_all.csv', 
                       n_samples = 30, 
                       spacing =2,
                       pca=True,
                       shuf=False)
                    
                       

plotting(metrics,metrics_rd,'./figs/pca.png')

#========================================================
# Testing model stability by running 100 case
# The data is shuffled at the beggining of each run 
# The run can take day(s), the results are in ./cases/shuffle_results.txt
summary = []
n_samples = [5,10,15,20,25,30]
#n_samples = [35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for n_samples in n_samples:
    lst = []
    lst_rand = []
    for i in range (1,101):
        metrics, metrics_rd = FE_AL(df = 'data_all.csv', 
                                    n_samples = n_samples, 
                                    spacing =5,
                                    pca=False,
                                    shuf=True) 
        lst.append([i,metrics[-1,1],metrics[-1,2]])
        lst_rand.append([i,metrics_rd[-1,1],metrics_rd[-1,2]])
        a = np.array(lst)
        a_rand = np.array(lst_rand)
        print("Number of samples =", n_samples)
        print ("i =",i,"/100")
        print(np.mean(a[:,1]),
              np.mean(a[:,2]), 
              np.mean(a_rand[:,1]), 
              np.mean(a_rand[:,2]))
    
    summary.append([n_samples, 
                    np.mean(a[:,1]),
                    np.std(a[:,1]),
                    np.mean(a[:,2]),
                    np.std(a[:,2]),
                    np.mean(a_rand[:,1]),
                    np.std(a_rand[:,1]),
                    np.mean(a_rand[:,2]),
                    np.std(a_rand[:,2])])
results = np.array(summary)
#np.savetxt('./cases/results.txt', results)
data = np.loadtxt('./cases/shuffle_results.txt')
metrics = data[:,0:3]
metrics_rand = data[:,(0,3,4)]
plotting(metrics,metrics_rand,'./figs/shuffle.png')

#===============================================================
# Using more than one GPR 
# RESULTS ARE UNSATISFACTORY 
# Uncomment and Check yourself
'''
metrics = ENAL(df='data_all.csv', 
               n_samples=30, 
               spacing=2, 
               pca=False, 
               shuf=False, 
               n_reg=8)
print(metrics)
'''