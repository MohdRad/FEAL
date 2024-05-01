# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:41:57 2024

@author: mirad
"""

from FEAL import FE_AL, plotting

metrics = FE_AL(df = 'data_all.csv', 
                       n_samples = 30, 
                       spacing =2, 
                       q_str = "max")

metrics_rd = FE_AL(df = 'data_all.csv', 
                       n_samples = 30, 
                       spacing =2, 
                       q_str = "random")


plotting(metrics,metrics_rd,'original.png')