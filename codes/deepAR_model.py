# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:54:55 2023

@author: gusta
"""

import pandas as pd
import numpy as np
import os

import memory_aux


import matplotlib.pyplot as plt
from gluonts.mx import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.pandas import PandasDataset


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def deepAR_MeltDF(train, valid, target_column, key_column, date_column, data_freq, prediction_length,
                  layers, cells, drop_rate, epochs, num_workers, lower_bound, higher_bound):
    
    print('Set data format')
    train_ds = PandasDataset.from_long_dataframe(train, target=target_column, item_id=key_column, 
                                           timestamp=date_column, freq=data_freq)
    print('Set Estimator')
    estimator = DeepAREstimator(freq=data_freq, prediction_length=prediction_length,
                                num_cells = cells, 
                                num_layers=layers, dropout_rate = drop_rate,
                                trainer=Trainer(#ctx = mx.context.gpu(),
                                                epochs=epochs, #learning_rate = learning_rate,
                                                #callbacks=[scheduler]
                                                ))
    print('Train Estimator')
    predictor = estimator.train(train_ds, num_workers=num_workers)
    print('Predicting Data not seen yet but using past data used on training')
    pred = list(predictor.predict(train_ds))
    
    all_preds = list()
    for item in pred:
        key = item.item_id
        p = item.samples.mean(axis=0)
        p_lower = np.percentile(item.samples, lower_bound, axis=0)
        p_higher = np.percentile(item.samples, higher_bound, axis=0)
        dates = pd.date_range(start=item.start_date.to_timestamp(), periods=len(p), freq = data_freq)
        family_pred = pd.DataFrame({date_column: dates, key_column: key, 'pred': p, 
                                    f'p_{lower_bound}': p_lower, f'p_{higher_bound}': p_higher})
        all_preds += [family_pred]
    all_preds = pd.concat(all_preds, ignore_index=True)
    
    all_preds = all_preds.set_index([date_column, key_column])
    all_preds[all_preds < 0] = 0
    all_preds = all_preds.reset_index()
    
    all_preds = all_preds.merge(valid, on=[date_column, key_column], how='left')
    all_preds['metric_Acc'] = 1 - (abs(all_preds[target_column] - all_preds['pred']
                                       )/all_preds[target_column])
    wmape_metric = wmape(all_preds[target_column], all_preds['pred'])
    
    print('Now predict using the valid set, that is the latest avaible data, to predict future')
    print('Predicting Data not realized yet')
    
    print('Set format')
    valid_ds = PandasDataset.from_long_dataframe(valid, target=target_column, item_id=key_column, 
                                           timestamp=date_column, freq=data_freq)
    pred_future = list(predictor.predict(valid_ds))
    
    all_preds_future = list()
    for item in pred_future:
        key = item.item_id
        p = item.samples.mean(axis=0)
        p_lower = np.percentile(item.samples, lower_bound, axis=0)
        p_higher = np.percentile(item.samples, higher_bound, axis=0)
        dates = pd.date_range(start=item.start_date.to_timestamp(), periods=len(p), freq = data_freq)
        family_pred = pd.DataFrame({date_column: dates, key_column: key, 'pred': p, 
                                    f'p_{lower_bound}': p_lower, f'p_{higher_bound}': p_higher})
        all_preds_future += [family_pred]
    all_preds_future = pd.concat(all_preds_future, ignore_index=True)
    
    all_preds_future = all_preds_future.set_index([date_column, key_column])
    all_preds_future[all_preds_future < 0] = 0
    all_preds_future = all_preds_future.reset_index()
    
    all_preds = memory_aux.reduce_mem_usage(all_preds)
    all_preds_future = memory_aux.reduce_mem_usage(all_preds_future)
    
    return all_preds, wmape_metric, all_preds_future
    
def plot_template_melt(df_long, all_preds, date_column, target_column, key_column,
                         lower_bound, higher_bound, final_path):
    
    df_group_vol = df_long.groupby([key_column]).sum([target_column]).sort_values(
        by = target_column, ascending = False)
    print('Plotting Predict and Real values for 6 keys with highest volumes in dataset')
    
    list_higher_vols = list( df_group_vol.head(6).index)
    fig, ax = plt.subplots(3,2, figsize=(1280/96, 720/96), dpi=96)
    ax = ax.flatten()
    for ax_ ,family in enumerate(list_higher_vols):
        p_ = all_preds.loc[all_preds[key_column] == family]
        p_.plot(x=date_column, y=target_column, ax=ax[ax_], label='Sales')
        p_.plot(x=date_column, y='pred', ax=ax[ax_], label='Forecast')
        ax[ax_].fill_between(p_[date_column].values, p_[f'p_{lower_bound}'], p_[f'p_{higher_bound}'], alpha=0.2, color='orange')
        ax[ax_].set_title(family)
        ax[ax_].legend()
        ax[ax_].set_xlabel(date_column)
        ax[ax_].set_ylabel(target_column)
        
    plt.savefig(os.path.join(final_path,'Results_High_Volume.png'))
    fig.tight_layout()
    
    return 'Plots Done'

def plot_hist_predict(valid, all_preds_future, df_long, target_column, date_column,
                      final_path, key_column):
    print('To check if predictions make sense, lets plot most recent history with next predictions')
    df_future = all_preds_future[[date_column, key_column, 'pred']]
    df_future = df_future.rename(columns = {'pred': target_column})
    df_final = pd.concat([valid, df_future], ignore_index=True)

    df_group_vol = df_long.groupby([key_column]).sum([target_column]).sort_values(
        by = target_column, ascending = False)
    print('Plotting Predict and Real values for 6 keys with highest volumes in dataset')
    
    list_higher_vols = list( df_group_vol.head(6).index)
    fig, ax = plt.subplots(3,2, figsize=(1280/96, 720/96), dpi=96)
    ax = ax.flatten()
    for ax_ ,family in enumerate(list_higher_vols):
        p_ = df_final.loc[df_final[key_column] == family]
        p_.plot(x=date_column, y=target_column, ax=ax[ax_], label='Sales')
        ax[ax_].axvline(pd.to_datetime(valid[date_column].max()), color='red', linestyle='--', label='Historic Limit')
        ax[ax_].set_title(family)
        ax[ax_].legend()
        ax[ax_].set_xlabel(date_column)
        ax[ax_].set_ylabel(target_column)
    plt.savefig(os.path.join(final_path,'History_and_Projections_High_Volume.png'))
    fig.tight_layout()
    
    
    return 'Plot Historic and Future Values'

if __name__ == '__main__':
    
    print('teste')
    
    
    