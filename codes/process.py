# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:43:15 2023

@author: gusta
"""

import pandas as pd
import numpy as np
import os
#get paths
general_path = r'C:\Users\gusta\Desktop\Personal_Projects'
project_path = general_path + r'\DeepAR_Template'
data_path = project_path + r'\data'
codes_path = project_path + r'\codes'
final_path = project_path + r'\final_files'
# get aux codes

os.chdir(codes_path)
import memory_aux
import data_prep
import deepAR_model


# configs
target_column = 'Weekly_Sales'
date_column = 'Date'
key_column = 'key_Store_Dept'
data_freq = 'W-FRI'
split_ratio = 0.95

learning_rate = 0.001
patience = 20
prediction_length = 8
context_length = 8
cells = 20
layers = 4
epochs = 20
drop_rate = 0.01
n_samples = 100
num_workers = 2
lower_bound = 20
higher_bound = 80


if __name__ == '__main__':
    
    print('Get data')
    df_univariate, split_date = data_prep.get_data(data_path, key_column, target_column, date_column, split_ratio)
    
    #####
    print('Set data ready for DeepAR')
    df_long, train, valid = data_prep.melt_df_back(df_univariate, split_date, date_column, key_column, target_column)
    
    print('Run and get results')
    all_preds, wmape_metric, all_preds_future = deepAR_model.deepAR_MeltDF(train, valid, target_column, key_column, date_column, data_freq, prediction_length,
                      layers, cells, drop_rate, epochs, num_workers, lower_bound, higher_bound)
    
    print('Clear folder with final files to save new files and plots')
    data_prep.clean_plots_files(final_path)
    
    all_preds.to_csv(os.path.join(final_path, 'results_test.csv'), sep = ';', decimal = '.')
    all_preds_future.to_csv(os.path.join(final_path, 'future_predictions.csv'), sep = ';', decimal = '.')
    
    print('Accuracy Overall weighted by Sales Volume', 1 - wmape_metric)

    #Plots
    deepAR_model.plot_template_melt(df_long, all_preds, date_column, target_column, key_column,
                             lower_bound, higher_bound, final_path)
    
    deepAR_model.plot_hist_predict(valid, all_preds_future, df_long, target_column, date_column,
                          final_path, key_column)

