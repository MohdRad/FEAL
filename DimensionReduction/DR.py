"""
Created on Fri Mar  8 10:47:19 2024
@author: Kumo
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class DRAL():
    
    '''
    Parameters: 
        path: (str) The path to your processed input (columns ready) 
    '''
        
    def __init__(self,path_X:str):
        self.X = pd.read_csv(path_X)
   
        
    def apply_scaling (self,X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    # Principal Component Analysis
    def PCA (self, var):
        '''
        Parameters
        ----------
        var : (float) Minimum variance to be retained

        '''
        # Original shape of X
        m,n = self.X.shape
        # Scale the data 
        X_scaled = self.apply_scaling(self.X)
        # Apply PCA
        pca = PCA()
        X_DR = pca.fit_transform(X_scaled)
        m_dr, n_dr = X_DR.shape
        # Calculate the cumulative explained variance
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        # Determine the number of components to keep for variance explained
        n_components = np.argmax(cumulative_variance_ratio >= var) + 1
        # Display the results
        print("Original Number of Features", n)
        print("Number of Components associated with the retained variance (95%):", n_components)
        # Plotting
        n_pca = np.linspace(1, n_dr, n_dr,dtype=int)
        fig,ax1 = plt.subplots()
        ax1.plot(n_pca,100*cumulative_variance_ratio,label='PCA')
        ax1.set_xlabel('Number of PCs')
        ax1.set_ylabel('% Variance Retained')
        ax1.legend()
        #plt.savefig('PCA_var.png', dpi=500)
        return X_DR[:,:n_components]
            
    def SVD (self,var):
        # Original shape of X
        m,n = self.X.shape
        # Scale the data 
        X_scaled = self.apply_scaling(self.X)
        # Apply SVD
        svd = TruncatedSVD(n_components=n) 
        X_DR = svd.fit_transform(X_scaled)
        m_dr, n_dr = X_DR.shape
        # Calculate the cumulative explained variance
        cumulative_variance_ratio = np.cumsum(svd.explained_variance_ratio_)
        # Determine the number of components to keep for variance explained
        n_components = np.argmax(cumulative_variance_ratio >= var) + 1
        # Display the results
        print("Original Number of Features", n)
        print("Reduced Number of Features:", n_dr)
        print("Number of Components associated with the retained variance:", n_components)
        # Plotting 
        n_svd = np.linspace(1, n_dr, n_dr,dtype=int)
        fig,ax1 = plt.subplots()
        ax1.plot(n_svd,100*cumulative_variance_ratio,label='SVD')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('% Variance Retained')
        ax1.legend()
        #plt.savefig('SVD_var.png', dpi=500)    
        return X_DR[:,:n_components]
    
    def ICA (self,var):
        # Original shape of X
        m,n = self.X.shape
        # Scale the data 
        X_scaled = self.apply_scaling(self.X)
        # Apply ICA
        ica = FastICA(n_components=n) 
        X_DR = ica.fit_transform(X_scaled)
        m_dr, n_dr = X_DR.shape
        # Calculate the cumulative explained variance
        explained_variance = np.var(X_DR, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        # Determine the number of components to keep for variance explained
        n_components = np.argmax(cumulative_variance_ratio >= var) + 1
        # Display the results
        print("Original Number of Features", n)
        print("Reduced Number of Features:", n_dr)
        print("Number of Components associated with the retained variance:", n_components)
        # Plotting
        n_ica = np.linspace(1, n_dr, n_dr,dtype=int)
        fig,ax1 = plt.subplots()
        ax1.plot(n_ica,100*cumulative_variance_ratio,label='ICA')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('% Variance Retained')
        ax1.legend()
        #plt.savefig('ICA_var.png', dpi=500) 
        return X_DR[:,:n_components]
        
        

    
    


