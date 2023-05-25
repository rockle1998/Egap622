# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 05:35:13 2022

@author: RockLee
"""

#----------------- Splitting data into the train/validation/test dataset -------------------#

#--------------------------------------------------------------------------#
#1) The libraries are used
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'
from sklearn.model_selection import train_test_split

RNG_SEED = 34
np.random.seed(seed=RNG_SEED)

#--------------------------------------------------------------------------#
#2) Loading data from file's csv(excel)
PATH =os.getcwd()                                           # Using the system
data_path = os.path.join(PATH, 'Egap/Egap_cleaned.csv')  # Looking for the location of file's in your computer
df = pd.read_csv(data_path)                                 # Reading the file's 
print(f'Full dataframe shape: {df.shape}')                  # print the data frame
print()

#3) Separate the Dataframe into your input variables (X)) and target variables (y))
X = df[['formula']] # Separate by dataframe variables (X[formula])) 
y = df['target']             # Separate by dataframe variables (target )) 
#print(f'Shape of X: {X.shape}')
#print(f'Shape of y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG_SEED) # using the train_test_split to separate the data frame
#print(X_train.shape)
#print(X_test.shape)

# check inique formula because  the same chemical compound appearing in both the training and test data. So how do we mitigate this?
num_rows = len(X_train) #length of X_train
print(f'There are in total {num_rows} (20%) rows in the X_train Dataframe.') # print total X_train
num_unique_formulae = len(X_train['formula'].unique()) #length of X_train[formula]
print(f'There are only {num_unique_formulae} unique formulae!\n')

print('Unique formulae and their number of occurances in the X_train data frame:')
print(X_train['formula'].value_counts(), '\n')
print('Unique formulae and their number of occurances in the X_test data frame:')
print(X_test['formula'].value_counts())
print()

#4) Splitting data, cautiously (manually)
unique_formulae = X['formula'].unique()
#print(f'{len(unique_formulae)} unique formulae: \n{unique_formulae}')

#a) set a random seed to ensure reproducibitity across run
np.random.seed(seed=RNG_SEED)

#b) Store all of unique_formulae
all_formulae = unique_formulae.copy()

#c) Define the percent for the model
val_size = 0.20 # 0.2 = 20%
test_size = 0.10 # 0.1 = 10%
train_size = 1 - val_size - test_size

#d) Calculate the number of samples in each dataset split 
num_val_samples = int(round(val_size * len(unique_formulae))) # The int() function converts a number or a string to its equivalent integer
num_test_samples = int(round(test_size * len(unique_formulae))) # The round() function returns a floating point number
num_train_smaples = int(round(train_size * len(unique_formulae))) # The len() function returns the number of items in an object.

#e) Randomly choose the formulae for the validaton and remove those from the unique_formulae list
val_formulae = np.random.choice(all_formulae, size=num_val_samples, replace=False) # random.choice(a, size=None, replace=True, p=None)
all_formulae = [f for f in all_formulae if f not in val_formulae] # when f in the all_formulae data frame and if not f in val_formulae

#f) Randomly choose the formulae for the test and remove those from the unique_formulae list
test_formulae = np.random.choice(all_formulae, size=num_test_samples, replace=False) # random.choice(a, size=None, replace=True, p=None)
all_formulae = [f for f in all_formulae if f not in test_formulae]    # when f in the all_formulae data frame and if not f in val_formulae

#g The remaining formula will be used for the training data set
train_formulae = all_formulae.copy()

print('Number training of formulae:', len(train_formulae))
print('Number validation of formulae:', len(val_formulae))
print('Number testing of formulae:', len(test_formulae))
print()

#5) Splitting original dataset into the test/validation/train 
df_train = df[df['formula'].isin(train_formulae)] # splitting df_train from train_formulae
df_test = df[df['formula'].isin(test_formulae)]   # splitting df_test from test_formulae
df_val = df[df['formula'].isin(val_formulae)]     # splitting df_val from val_formulae

print(f'Training dataset shape: {df_train.shape}')
print(f'Testing dataset shape: {df_test.shape}')
print(f'Validation dataset shape: {df_val.shape}')
print()
#printing table of dataset shape
#print(df_train.head(), '\n')
#print(df_test.head, '\n')
#print(df_val.head, '\n')
# check the data have mutually exclusive formulae
train_formulae = set(df_train['formula'].unique()) # The set() function creates a set object.
test_formulae = set(df_test['formula'].unique())   # The unique() function is used to find the unique elements of an array.
val_formulae = set(df_val['formula'].unique())

common_formulae1 = train_formulae.intersection(test_formulae)
common_formulae2 = train_formulae.intersection(val_formulae)
common_formulae3 = test_formulae.intersection(val_formulae)

print(f'# of common formulae in intersection 1: {len(common_formulae1)}; common formulae: {common_formulae1}')
print(f'# of common formulae in intersection 2: {len(common_formulae2)}; common formulae: {common_formulae2}')
print(f'# of common formulae in intersection 3: {len(common_formulae3)}; common formulae: {common_formulae3}')

#6) Saving the dataset 
PATH = os.getcwd()

train_path = os.path.join(PATH, 'data_split/train.csv')
test_path = os.path.join(PATH, 'data_split/test.csv')
val_path = os.path.join(PATH, 'data_split/val.csv')

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)
df_val.to_csv(val_path, index=False)

print("------------------------END------------------------")








