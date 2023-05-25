# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:14:23 2022

@author: RockLee
"""
#------------------- Data loading, cleanup and processing -----------------------------------#
#1) The libraries are used 
import os
import numpy as np
import pandas as pd

#--------------------------------------------------------------------------#
#2)loading data
PATH = os.getcwd()
data_path = os.path.join(PATH,'Egap/Egap.csv')

df = pd.read_csv(data_path)
print(f'Original Dataframe Shape: {df.shape}')
print()
# print(df.head(10))
print(df.loc[291], '\n')
print(f'Summary statistics of the dataframe: {df.describe()}')

#--------------------------------------------------------------------------#
#3) check for and remove NaN and unrealistic values
#   a) check for and remove NaN
df2 = df.copy()
bool_nan_formula = df2['formula'].isnull()
bool_nan_target = df2['target'].isnull()

df2 = df2.drop(df2.loc[bool_nan_formula].index, axis = 0)
df2 = df2.drop(df2.loc[bool_nan_target].index, axis = 0)
print()
print(f'Dataframe shape before dropping Nans: {df.shape}')
print(f'Dataframe shape after dropping Nans: {df2.shape}')

#   b) check for and unrealistics
df = df2.copy()
bool_invalid_target_0 = df['target'] <= 0 
#bool_invalid_target = df['target'] >= 10.0

df = df.drop(df.loc[bool_invalid_target_0].index, axis = 0)
#df = df.drop(df.loc[bool_invalid_target].index, axis = 0)
print()
print(f'Cleaned Dataframe shape of: {df.shape}')

#--------------------------------------------------------------------------#
#4) Saving cleaned data to csv
out_path = os.path.join(PATH,'Egap/Egap_cleaned.csv')
df.to_csv(out_path, index = False)

#--------------------------------------------------------------------------#

#Ending the coding  
print()
print("------------------------END------------------------")























