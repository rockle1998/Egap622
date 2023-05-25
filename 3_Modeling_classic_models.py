# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 00:30:04 2023

@author: ROCKLE
"""
#------------------- Modeling classic models -----------------------------------#
# we will show some simple examples of featurizing materials composition data using so-called "composition-based feature vectors", or CBFVs. 
#This methods represents a single chemical formula as one vector based on its constituent atoms' chemical properties (refer to the paper for more information and references).

#--------------------------------------------------------------------------#
#1) The libraries are used 
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'

from collections import OrderedDict

RNG_SEED = 34
np.random.seed(RNG_SEED)

#--------------------------------------------------------------------------#
#2) Reading the file csv
#a) reading the file csv
PATH = os.getcwd()
train_path = os.path.join(PATH, 'data_split/train.csv')
test_path = os.path.join(PATH, 'data_split/test.csv')
val_path = os.path.join(PATH, 'data_split/val.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_val = pd.read_csv(val_path)
print(f'Dataframe for modeling classic model')
print(f'df_train dataframe of shape: {df_train.shape}')
print(f'df_test dataframe of shape: {df_test.shape}')
print(f'df_val dataframe of shape: {df_val.shape}')
print()

#b)sub _sampling your data as suitable for the modeling 
df_train_sampled = df_train.sample(n=2000, random_state=RNG_SEED) # random train with n =2000 in file train (3214)
df_test_sampled = df_test.sample(n=200, random_state=RNG_SEED)    # random test with n =200 in file test (370)
df_val_sampled = df_val.sample(n=200, random_state=RNG_SEED)      # random val with n =200 in file val (980)

print(f'Sub_sampling dataframe shape for modeling classic model')
print(f'df_train_sampled dataframe shape: {df_train_sampled.shape}')
print(f'df_test_sampled dataframe shape: {df_test_sampled.shape}')
print(f'df_val_sampled dataframe shape: {df_val_sampled.shape}')

print()
print(f'Generate features using the CBFV package')

#--------------------------------------------------------------------------#
#3) Generate features using CBFV: composition-based feature vectors"
from CBFV import composition 
# the library, Tool to quickly create a composition-based feature vectors from materials datafiles.

print('DataFrame column names before renaming:')
print(df_train.columns)
print(df_val.columns)
print(df_test.columns)

rename_dict = {'Egap': 'target'} # to choose the Cp to calculate in the model classcal
df_train = df_train.rename(columns=rename_dict) 
df_val = df_val.rename(columns=rename_dict)
df_test = df_test.rename(columns=rename_dict)

print('\nDataFrame column names after renaming:')
df_train_sampled = df_train_sampled.rename(columns=rename_dict) # this is the sample for model with n which is chose
df_val_sampled = df_val_sampled.rename(columns=rename_dict)
df_test_sampled = df_test_sampled.rename(columns=rename_dict)
print(df_train.columns)
print(df_val.columns)
print(df_test.columns)
print()

# Trainning the CBFV with the train/test/val 
# X, y, formulae, skipped = composition.generate_features(df)
X_train_unscaled, y_train, formulae_train, skipped_train = composition.generate_features(df_train_sampled, elem_prop='oliynyk', 
                                                                                         drop_duplicates=False, extend_features=True, 
                                                                                         sum_feat=True)

X_val_unscaled, y_val, formulae_val, skipped_val = composition.generate_features(df_val_sampled, elem_prop='oliynyk', 
                                                                                 drop_duplicates=False, extend_features=True, 
                                                                                 sum_feat=True)

X_test_unscaled, y_test, formulae_test, skipped_test = composition.generate_features(df_test_sampled, elem_prop='oliynyk', 
                                                                                     drop_duplicates=False, extend_features=True, 
                                                                                     sum_feat=True)

print(X_train_unscaled.shape)

#--------------------------------------------------------------------------#
#4)Data scaling & normalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

scaler = StandardScaler() #StandardScaler(*, copy=True, with_mean=True, with_std=True) 

X_train = scaler.fit_transform(X_train_unscaled) # Fit to data, then transform it.
X_val = scaler.transform(X_val_unscaled)
X_test = scaler.transform(X_test_unscaled)

X_train = normalize(X_train) #The normalize() function scales vectors individually to a unit norm so that the vector has a length of one.
X_val = normalize(X_val)
X_test = normalize(X_test)


#--------------------------------------------------------------------------#
#5)Modeling using "classical" machine learning models
print()
print(f'Modeling using "classical" machine learning models')
# using the model from the classical 
from time import time

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# define the value for model: instantiate, fitting, evaluate, fitting evaluate, append results model 
def instantiate_model(model_name):
    model = model_name()
    return model

def fit_model(model, X_train, y_train): 
    ti = time()
    model = instantiate_model(model)
    model.fit(X_train, y_train)
    fit_time = time() - ti
    return model, fit_time

def evaluate_model(model, X, y_act):
    y_pred = model.predict(X)
    r2 = r2_score(y_act, y_pred)
    mae = mean_absolute_error(y_act, y_pred)
    rmse_val = mean_squared_error(y_act, y_pred, squared=False)
    return r2, mae, rmse_val

def fit_evaluate_model(model, model_name, X_train, y_train, X_val, y_act_val):
    model, fit_time = fit_model(model, X_train, y_train)
    r2_train, mae_train, rmse_train = evaluate_model(model, X_train, y_train)
    r2_val, mae_val, rmse_val = evaluate_model(model, X_val, y_act_val)
    result_dict = {
        'model_name': model_name,
        'model_name_pretty': type(model).__name__,
        'model_params': model.get_params(),
        'fit_time': fit_time,
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val}
    return model, result_dict

def append_result_df(df, result_dict):
    df_result_appended = df.append(result_dict, ignore_index=True)
    return df_result_appended

def append_model_dict(dic, model_name, model):
    dic[model_name] = model
    return dic

# the name of classics columns 
df_classics = pd.DataFrame(columns=['model_name',
                                    'model_name_pretty',
                                    'model_params',
                                    'fit_time',
                                    'r2_train',
                                    'mae_train',
                                    'rmse_train',
                                    'r2_val',
                                    'mae_val',
                                    'rmse_val'])

#--------------------------------------------------------------------------#
#a) Define the models
# Build a dictionary of model names, Instantiate and fit the models
classic_model_names = OrderedDict({
    'dumr': DummyRegressor,
    'rr': Ridge,
    'abr': AdaBoostRegressor,
    'gbr': GradientBoostingRegressor,
    'rfr': RandomForestRegressor,
    'etr': ExtraTreesRegressor,
    'svr': SVR,
    'lsvr': LinearSVR,
    'knr': KNeighborsRegressor,
})

# Instantiate a dictionary to store the model objects
classic_models = OrderedDict()

# Keep track of elapsed time
ti = time()

# Loop through each model type, fit and predict, and evaluate and store results
for model_name, model in classic_model_names.items():
    print(f'Now fitting and evaluating model {model_name}: {model.__name__}')
    
    model, result_dict = fit_evaluate_model(model, model_name, X_train, y_train, X_val, y_val)
    
    df_classics = append_result_df(df_classics, result_dict)
    
    classic_models = append_model_dict(classic_models, model_name, model)

dt = time() - ti
print(f'Finished fitting {len(classic_models)} models, total time: {dt:0.2f} s')


# Sort in order of increasing validation r2 score
df_classics = df_classics.sort_values('r2_val', ignore_index=True)


#--------------------------------------------------------------------------#
#c)Evaluating model performance on validation dataset.
   # In addition, we plot the predicted vs. actual plots using the predictions made by each trained model on the same validation set.

def plot_pred_act(act, pred, model, reg_line=True, label=''):
    xy_max = np.max([np.max(act), np.max(pred)])

    plot = plt.figure(figsize=(10,6))
    plt.plot(act, pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)
    plt.plot([0, xy_max], [0, xy_max], 'k--', label='ideal')
    if reg_line:
        polyfit = np.polyfit(act, pred, deg=1)
        reg_ys = np.poly1d(polyfit)(np.unique(act))
        plt.plot(np.unique(act), reg_ys, alpha=0.8, label='linear fit')
    plt.axis('scaled')
    plt.xlabel(f'Actual {label}', size = 14)
    plt.ylabel(f'Predicted {label}', size = 14)
    plt.title(f'{type(model).__name__}, r2: {r2_score(act, pred):0.4f}')
    plt.legend(loc='upper left')
    
    return plot

for row in range(df_classics.shape[0]):
    model_name = df_classics.iloc[row]['model_name']

    model = classic_models[model_name]
    y_act_val = y_val
    y_pred_val = model.predict(X_val)

    plot = plot_pred_act(y_act_val, y_pred_val, model, reg_line=True, label='Egap (eV)')

#--------------------------------------------------------------------------#
#d) Re-training the best-performing model on combined train + validation dataset

print()
# Find the best-performing model that we have tested
best_row = df_classics.iloc[-1, :].copy()

# Get the model type and model parameters
model_name = best_row['model_name']
model_params = best_row['model_params']

# Instantiate the model again using the parameters
model = classic_model_names[model_name](**model_params)
print(f'Find the best-performing model: {model}')

# Concatenate the train and validation datasets together
X_train_new = np.concatenate((X_train, X_val), axis=0)
y_train_new = pd.concat((y_train, y_val), axis=0)

print(f'Concatenate the train and validation dataset: {X_train_new.shape}')

ti = time()

model.fit(X_train_new, y_train_new)

dt = time() - ti
print(f'Finished fitting best model, total time: {dt:0.2f} s') #Is the format to print just 2 decimals instead of all of them when you print a float5

#e)Testing the re-trained model on the test dataset
y_act_test = y_test
y_pred_test = model.predict(X_test)

r2, mae, rmse = evaluate_model(model, X_test, y_test)
print(f'r2: {r2:0.4f}')
print(f'mae: {mae:0.4f}')
print(f'rmse: {rmse:0.4f}')

plot = plot_pred_act(y_act_test, y_pred_test, model, reg_line=True, label='Egap (eV)')
print()


#--------------------------------------------------------------------------#
#6) Effect of train/validation/test dataset split 
X_train_unscaled, y_train, formulae_train, skipped_train = composition.generate_features(df_train, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)
X_val_unscaled, y_val, formulae_val, skipped_val = composition.generate_features(df_val, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)
X_test_unscaled, y_test, formulae_test, skipped_test = composition.generate_features(df_test, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)

X_train_original = X_train_unscaled.copy()
X_val = X_val_unscaled.copy()
X_test = X_test_unscaled.copy()

y_train_original = y_train.copy()

splits = range(10)
df_splits = pd.DataFrame(columns=['split',
                                  'r2_train',
                                  'mae_train',
                                  'rmse_train',
                                  'r2_val',
                                  'mae_val',
                                  'rmse_val'])

for split in splits:
    print(f'Fitting and evaluating random split {split}')
    X_train = X_train_original.sample(frac=0.7, random_state=split) #The examples explained here will help you split the pandas DataFrame into two random samples (80% and 20%) for training and testing
    y_train = y_train_original[X_train.index] 

    scaler = StandardScaler()
    X_train = normalize(scaler.fit_transform(X_train))
    X_val = normalize(scaler.transform(X_val_unscaled))
    X_test = normalize(scaler.transform(X_test_unscaled))
    
    model = AdaBoostRegressor()
    model.fit(X_train, y_train)
    y_act_val = y_val
    y_pred_val = model.predict(X_val)

    r2_train, mae_train, rmse_train = evaluate_model(model, X_train, y_train)
    r2_val, mae_val, rmse_val = evaluate_model(model, X_val, y_val)
    result_dict = {
        'split': split,
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val}
    
    df_splits = append_result_df(df_splits, result_dict)

df_splits['split'] = df_splits['split'].astype(int)
print()
print(f'Random splits and evaluates' )
print(df_splits)

df_splits.plot('split', ['r2_train', 'r2_val'], kind='bar')
plt.title(f'Performance of {type(model).__name__}\nwith {len(splits)} different data splits')
plt.ylim((0, 1.0))
plt.ylabel('$r^2$', size = 14)
plt.xlabel('Split #', size = 14)
plt.legend(loc='upper right', framealpha=0.9)
plt.show()

df_splits.plot('split', ['mae_train', 'mae_val'], kind='bar')
plt.title(f'Performance of {type(model).__name__}\nwith {len(splits)} different data splits')
plt.ylabel('MAE in Egap (eV)', size = 14)
plt.xlabel('Split #', size = 14)
plt.legend(loc='lower right', framealpha=0.9)
plt.show()

df_splits.plot('split', ['rmse_train', 'rmse_val'], kind='bar')
plt.title(f'Performance of {type(model).__name__}\nwith {len(splits)} different data splits')
plt.ylabel('RMSE in Egap (eV)', size = 14)
plt.xlabel('Split #', size = 14)
plt.legend(loc='lower right', framealpha=0.9)
plt.show()

avg_r2_val = df_splits['r2_val'].mean()
avg_mae_val = df_splits['mae_val'].mean()
avg_rmse_val = df_splits['rmse_val'].mean()
print()
print(f'Average validation r2: {avg_r2_val:0.4f}')
print(f'Average validation MAE: {avg_mae_val:0.4f}')
print(f'Average validation RSME: {avg_rmse_val:0.4f}')

PATH = os.getcwd()
evaluating_path = os.path.join(PATH,'random_splits_and_evaluate/random_splits_and_evaluate_with_seed_34.csv')
df_splits.to_csv(evaluating_path, index=False)

print()
print(f'------------------------END------------------------')