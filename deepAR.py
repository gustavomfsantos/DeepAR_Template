# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:59:43 2023

@author: gusta
"""
#About Features_DF


# Features
# Contains additional data related to the store, department, and regional 
#activity for the given dates.

# Store - the store number
# Date - the week
# Temperature - average temperature in the region
# Fuel_Price - cost of fuel in the region
# MarkDown 1-5 - anonymized data related to promotional markdowns. MarkDown data 
# is only available after Nov 2011, and is not available for all stores all 
#the time. Any missing value is marked with an NA
# CPI - the consumer price index
# Unemployment - the unemployment rate
# IsHoliday - whether the week is a special holiday week

#About Markdown
# The company also runs several promotional markdown events throughout the year.
# These markdowns precede prominent holidays, 
# the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and 
#Christmas. The weeks including these holidays are 
# weighted five times higher in the evaluation than non-holiday weeks.


# Sales
# Historical sales data, which covers to 2010-02-05 to 2012-11-01. Within this 
#tab you will find the following fields:

# Store - the store number
# Dept - the department number
# Date - the week
# Weekly_Sales -  sales for the given department in the given store
# IsHoliday - whether the week is a special holiday week

# Stores
# Anonymized information about the 45 stores, indicating the type and size of store

#Getting DataSet and Prep Data

import pandas as pd
import numpy as np
import os
# from sklearn.preprocessing import OneHotEncoder
data_path =  r"C:\Users\gusta\Desktop\Personal_Projects\Sales_Project_Data"

print(os.listdir(data_path))

features_df = pd.read_csv(os.path.join(data_path, 'Features data set.csv'))
sales_df = pd.read_csv(os.path.join(data_path, 'sales data-set.csv'))
stores_df = pd.read_csv(os.path.join(data_path, 'stores data-set.csv'))

#Store Type Is related to size.
#Instead of get dummies for each store, create a column where
#biggest store gets highernumber and smaller store gets lower number
stores_df['Size_Type'] = np.where(stores_df['Type'] == 'A', 3,
                                  np.where(stores_df['Type'] == 'B', 2 ,1))
stores_df = stores_df.drop(['Type'], axis = 1)



features_df_ = pd.merge(features_df, stores_df, how = 'left', on = ['Store'])
print('Ajust NAN')
features_df_['Date'] = pd.to_datetime(features_df_['Date'], format='%d/%m/%Y')
# Check Date
features_df_[[ 'MarkDown1', 'MarkDown2',
       'MarkDown3', 'MarkDown4', 'MarkDown5']] = features_df_[[ 'MarkDown1', 
                                                               'MarkDown2',
       'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
features_df_['IsHoliday'] =  np.where(features_df_['IsHoliday'] == True, 1, 0)                                                              
features_df_['year_month'] =  (features_df_['Date'].astype(str).str[:4] + 
                               features_df_['Date'].astype(str).str[5:7]   )    

features_df__ = features_df_.drop(['Date'], axis = 1
    ).groupby(['year_month', 'Store']).apply(lambda x: x.mean()).drop(
        ['Store', 'year_month'], axis = 1).reset_index()  
features_df__['IsHoliday'] =  np.where(features_df__['IsHoliday'] >= 0, 1, 0)                     
print('Most Recent data not avaible for CPI and Unployment')
# features_df_.info()
# x = features_df_.describe()

#Change Is Holiday for dummy

# sales_df['Holiday_Dummy'] = np.where(sales_df['IsHoliday'] == True, 1, 0)
sales_df = sales_df.drop(['IsHoliday'], axis  = 1)
sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d/%m/%Y')
sales_df['year_month'] =  (sales_df['Date'].astype(str).str[:4] + 
                               sales_df['Date'].astype(str).str[5:7]   )    

sales_df_ = sales_df.drop(['Date'], axis = 1
    ).groupby(['year_month', 'Store', 'Dept']).apply(lambda x: x.sum()).drop(
        ['Store', 'year_month', 'Dept'], axis = 1).reset_index().rename(
            columns = {'Weekly_Sales': 'Sales'})

df_sales = pd.merge(sales_df_, features_df__, how = 'left', on = ['year_month',
                                                                  'Store'])

del features_df_, features_df__, stores_df, sales_df,sales_df_, features_df



print('Data Prep')
print('Create key for Store-Department')
df_sales['key_Store_Dept'] = df_sales['Store'].astype(str) + '-' +  df_sales[
    'Dept'].astype(str) 

print('Transform year_month into date column')
df_sales['Date'] = pd.to_datetime( df_sales['year_month'].astype(str) + '01')
df_sales.drop(['year_month', 'Store', 'Dept'], axis = 1, inplace = True)

print('rearrange columns order, can even delete the drop command used previously')
df_sales = df_sales[['Date', 'key_Store_Dept', 'Sales', 'Temperature',
                     'Fuel_Price', 'MarkDown1', 'MarkDown2',
       'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
       'IsHoliday', 'Size', 'Size_Type']]


print('Check Data Avaible for each store-dept')
print('Get combination with 5 or less observations')
# df_few_obs = pd.DataFrame(columns = ['Store_Dept', 'Number_obs'])
# for store_id in df_sales['key_Store_Dept'].unique():
#     print(store_id, len(df_sales[df_sales['key_Store_Dept'] == store_id]))
#     if len(df_sales[df_sales['key_Store_Dept'] == store_id]) <6:
#         list_append = [store_id, len(df_sales[df_sales['key_Store_Dept'] == store_id])]
#         df_few_obs.loc[len(df_few_obs)] = list_append
# del list_append, store_id
# print(len(df_few_obs), 'Combinations with less than 6 observations')

#DeepAR funciona com varias séries temporais juntas.
#Coluna Key distingue as séries
#Multiple time series in DeepAR - Date as index and time series as columns
print('Pivot table')
df_sales_pivot = pd.pivot_table(data = df_sales, columns = 'key_Store_Dept',
                                index = 'Date', values = 'Sales').dropna(thresh=25, axis=1)



from gluonts.dataset.common import ListDataset





#Try another approach

# target = '1-2'
# df_sales_pivot1 = df_sales_pivot[[target]]
# df_sales_pivot1.plot()

# Assuming you have a list of time series data, each with its own target and start date
# Example: time_series_data = [(target1, start1), (target2, start2), ...]



split_date = pd.Timestamp("2012-07-01")  # Replace "yyyy-mm-dd" with your desired split date

# Alternatively, for a percentage split
train_ratio = 0.9  # 80% of the data for training, 20% for testing
split_index = int(len(df_sales_pivot) * train_ratio)


train_data = df_sales_pivot[df_sales_pivot.index < split_date]  # or sales_data[:split_index] for percentage split
test_data = df_sales_pivot[df_sales_pivot.index >= split_date]  # or sales_data[split_index:] for percentage split

time_series_data = []
for column in train_data.columns:
    target = train_data[column].values
    start = train_data.index[0]  # Assuming the index of the DataFrame represents the time points
    time_series_data.append({"target": target, "start": start})


time_series_data_test = []
for column in test_data.columns:
    target = test_data[column].values
    start = test_data.index[0]  # Assuming the index of the DataFrame represents the time points
    time_series_data_test.append({"target": target, "start": start})


from gluonts.dataset.common import ListDataset


train_ds = ListDataset(time_series_data, freq="M")
test_ds = ListDataset(time_series_data_test, freq="M")
# train_ds = ListDataset(
#     [{"start": train_data.index.min(), "target": train_data[target].values}],
#     freq="M"
# )

# test_ds = ListDataset(
#     [{"start": test_data.index.min(), "target": test_data[target].values}],
#     freq="M"
# )


from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt




import mxnet as mx
from gluonts.mx import DeepAREstimator
from gluonts.mx.trainer import Trainer

# callbacks = [
#     LearningRateReduction(objective="min",
#                           patience=10,
#                           base_lr=1e-3,
#                           decay_factor=0.5,
#                           ),
#     ModelAveraging(avg_strategy=SelectNBestMean(num_models=2))
# ]

#Another callback - early stopper epochs
from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
learning_rate = 0.001

scheduler = LearningRateReduction(patience=20, base_lr=learning_rate, objective='min')

estimator = DeepAREstimator(
    freq="M",
    prediction_length=3,
    context_length=28,
    num_cells= 40,
    num_layers= 4,
    #distr_output=StudentTOutput(),
    dropout_rate=0.01,
    
    
    trainer=Trainer(#ctx = mx.context.gpu(),
                    epochs=20, learning_rate = learning_rate,
                    callbacks=[scheduler])) #callbacks=callbacks

predictor = estimator.train(train_ds)


from gluonts.evaluation.backtest import make_evaluation_predictions


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)
#tss = tss[0].iloc[:len(forecasts[0].index)]

for i in range(len(forecasts)):
    plt.figure(figsize=(10, 6))
    forecast_entry = forecasts[i]
    ts_entry = tss[i]
    
    # Plot the ground truth
    plt.plot(ts_entry[i].values[:-1], label="Ground Truth", 
             )
    
    # Plot the median forecast
    plt.plot(forecast_entry.median, label="Median Forecast",
             )
    
    # Plot the 90% confidence interval
    plt.fill_between(
        (np.array( range(0,len(ts_entry[i].values[:-1])))),
        # (str((forecast_entry.index.year)) +
        # str( forecast_entry.index.month)), #forecast_entry.index,
        forecast_entry.quantile(0.1),
        forecast_entry.quantile(0.9),
        color="lightblue",
        alpha=0.5,
        label="90% Confidence Interval"
    )
    
    plt.legend()
    plt.grid()
    plt.show()




from gluonts.evaluation import Evaluator

evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))

import json
print(json.dumps(agg_metrics, indent=4))





















