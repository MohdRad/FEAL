import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data_all.csv')

sample, test = train_test_split(df, 
                                test_size=0.32, 
                                random_state=42)


ltr = len(sample)
lte = len(test)
df = pd.concat([sample,test], axis=0)
cols = df.columns
letters = df['letters']
letters = letters.sort_values(ascending=True)
letters_arr = np.array(letters)
a = np.array(letters[:42])
df_arr = np.array(df)
sample_list = []
test_list = []

for i in range (len(df_arr)):
    if (df_arr[i,310] in a):
        test_list.append(df_arr[i])
        pass
    else:
        sample_list.append(df_arr[i])
    
samples = np.array(sample_list)
tests = np.array(test_list)

df_new = np.concatenate((samples,tests), axis=0)
df_new = pd.DataFrame(df_new, columns=cols)
df_new.to_csv('./cases/noA.csv',index=False)


'''
homo = ['AA', 'BB', 'CC', 
        'DD', 'EE', 'FF', 
        'GG', 'HH', 'JJ', 
        'KK', 'PP', 'QQ', 
        'RR', 'SS']
homo_lst = []
for i in range (len(df_arr)):
    if (df_arr[i,310] in hommo):
        hommo_list.append(df_arr[i])
        pass

homos = pd.DataFrame(homo_lst, columns=cols)
hommous = homos[['letters','arom_rings']]
hommous['letters'] = ['A','B','C', 'D',
                      'E','F','G',
                      'H','J','K',
                      'P','Q','R','S']
hommous = hommous.drop_duplicates()
hommous = hommous.sort_values(ascending=True, by='letters')
plt.rcParams.update({'font.size': 14})
hommous.plot.bar(x='letters', 
                 y='arom_rings', 
                 xlabel='PAC',
                 ylabel='Number of Aromatic Rings',
                 rot=0,
                 legend=False,
                 color=['tab:blue', 'tab:blue',
                        'tab:red', 'tab:red','tab:red', 
                        'tab:blue',
                        'tab:red','tab:red',
                        'tab:blue',
                        'tab:blue', 'tab:green',
                        'tab:green', 'tab:red','tab:red'])
plt.ylim((0,20))
plt.yticks([0,5,10,15,20])
plt.savefig('pacs.png',dpi=500, bbox_inches='tight')
'''