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

start_egap = 0 #(eV)
end_egap = 6   #(eV)

#--------------------------------------------------------------------------#
#2)loading data
PATH = os.getcwd()
data_path = os.path.join(PATH,'Egap/Egap.csv')

df = pd.read_csv(data_path)
print(f'Original Dataframe Shape: {df.shape}')
print()
#print(df.head(10))
#print()
#print(df.loc[10], '\n')
print(df.columns)
rename_dict = {'target': 'Egap'}
df = df.rename(columns=rename_dict)
print(df.columns)
print()
#--------------------------------------------------------------------------#
#3) check for and remove NaN and unrealistic values
df1 = df.copy()
df1 = df1.dropna(axis=0, how='any')
print(f'Dataframe shape before dropping NaNs: {df.shape}')
print(f'Dataframe shape after dropping NaNs: {df1.shape}')

df2=df1.copy()
bool_invalid_egap = df2['Egap'] <= start_egap
df2 = df2.drop(df2.loc[bool_invalid_egap].index, axis = 0)

df=df2.copy()
bool_invalid_egap = df['Egap'] >= end_egap
df = df.drop(df.loc[bool_invalid_egap].index, axis = 0)

print()
print(f'Summary statistics of the dataframe: {df.describe()}')
print()
print(f'Cleaned dataframe shape: {df.shape}')
#--------------------------------------------------------------------------#
#4) Saving cleaned data to csv
out_path = os.path.join(PATH,'Egap/Egap_cleaned.csv')
df.to_csv(out_path, index = False)

#--------------------------------------------------------------------------#

#Ending the coding  
print()
print("------------------------END------------------------")


















