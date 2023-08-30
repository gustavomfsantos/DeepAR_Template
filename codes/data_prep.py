# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:40:50 2023

@author: gusta
"""

'''
About Features_DF

Features
Contains additional data related to the store, department, and regional 
activity for the given dates.

Store - the store number
Date - the week
Temperature - average temperature in the region
Fuel_Price - cost of fuel in the region
MarkDown 1-5 - anonymized data related to promotional markdowns. MarkDown data 
is only available after Nov 2011, and is not available for all stores all 
the time. Any missing value is marked with an NA
CPI - the consumer price index
Unemployment - the unemployment rate
IsHoliday - whether the week is a special holiday week

About Markdown
The company also runs several promotional markdown events throughout the year.
These markdowns precede prominent holidays, 
the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and 
Christmas. The weeks including these holidays are 
weighted five times higher in the evaluation than non-holiday weeks.


Sales
Historical sales data, which covers to 2010-02-05 to 2012-11-01. Within this 
tab you will find the following fields:

Store - the store number
Dept - the department number
Date - the week
Weekly_Sales -  sales for the given department in the given store
IsHoliday - whether the week is a special holiday week

Stores
Anonymized information about the 45 stores, indicating the type and size of store
'''


#Get libs
import pandas as pd
import numpy as np
import os
from datetime import timedelta

import memory_aux


def import_dataset(data_path):
    print('Files')
    print(os.listdir(data_path))

    features_df = pd.read_csv(os.path.join(data_path, 'Features data set.csv'))
    sales_df = pd.read_csv(os.path.join(data_path, 'sales data-set.csv'))
    stores_df = pd.read_csv(os.path.join(data_path, 'stores data-set.csv'))
    print('Data loaded')
    print('Reduce memory use')
    features_df = memory_aux.reduce_mem_usage(features_df)
    sales_df = memory_aux.reduce_mem_usage(sales_df)
    stores_df = memory_aux.reduce_mem_usage(stores_df)
    print('Memory use Reduced ')
    #Store Type Is related to size.
    #Instead of get dummies for each store, create a column where
    #biggest store gets highernumber and smaller store gets lower number
    #So it brings ordinal order to store size instead of inly dummies
    stores_df['Size_Type'] = np.where(stores_df['Type'] == 'A', 3,
                                      np.where(stores_df['Type'] == 'B', 2 ,1))
    stores_df = stores_df.drop(['Type'], axis = 1)
    print('Size Store ordinal feature created')
    
    #Merge store information on features_df. Features_df contains information about
    #Each store. But the sales is separated by departments inside each store
    features_df = pd.merge(features_df, stores_df, how = 'left', on = ['Store'])
    features_df['Date'] = pd.to_datetime(features_df['Date'], format='%d/%m/%Y')
    print('Store features included on Sales Dataset')
    # Check Date
    #Markdown is features related to promotions and discount but without 
    #revelated wich promo is
    features_df[[ 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5']] = features_df[[ 'MarkDown1', 
                                                                   'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
    print('Ajusted NAN for markdown')                                                  
    #make column true - false into binary dummy
    features_df['IsHoliday'] =  np.where(features_df['IsHoliday'] == True, 1, 0) 
    
    #Creating columns with month and year. But data freq is weekly                                                             
    features_df['year_month'] =  (features_df['Date'].astype(str).str[:4] + 
                                   features_df['Date'].astype(str).str[5:7]   )    

    print('Most Recent data not avaible for CPI and Unployment')
    
    sales_df = sales_df.drop(['IsHoliday'], axis  = 1)
    sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d/%m/%Y')
    print('Ajusted date format for sales data and droped feature duplicated')

    df_sales_final = pd.merge(sales_df, features_df, how = 'left', on = ['Date',
                                                                      'Store'])
    del stores_df, sales_df,features_df
    print('Final merge realized and dataframes deleted')
    
    print('Create key for Store-Department')
    df_sales_final['key_Store_Dept'] = df_sales_final['Store'].astype(str) + '_' +  df_sales_final[
        'Dept'].astype(str) 
    
    df_sales_final.sort_values(['Store', 'Dept', 'Date'], inplace = True)
    df_sales_final.drop(['Store', 'Dept'], axis = 1, inplace = True)
    df_sales_final.reset_index(drop = False, inplace = True)    
    print('Values sorted by store, dept and date')

    df_sales_final = memory_aux.reduce_mem_usage(df_sales_final)

    df_sales_final = df_sales_final[['Date', 'key_Store_Dept', 'Weekly_Sales', 'Temperature',
                         'Fuel_Price', 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
           'IsHoliday', 'Size', 'Size_Type']]
    
    print('Final Dataset is ready!')
    
    return df_sales_final
    


def count_obs_for_key(df_sales_final):
    
    print('Check Data Avaible for each store-dept')
    print('Get combination with 5 or less observations')
    df_few_obs = pd.DataFrame(columns = ['Store_Dept', 'Number_obs'])
    for store_id in df_sales_final['key_Store_Dept'].unique():
        print(store_id, len(df_sales_final[df_sales_final['key_Store_Dept'] == store_id]))
        # if len(df_sales_final[df_sales_final['key_Store_Dept'] == store_id]) <6:
        list_append = [store_id, len(df_sales_final[df_sales_final['key_Store_Dept'] == store_id])]
        df_few_obs.loc[len(df_few_obs)] = list_append
    del list_append, store_id
    print(len(df_few_obs), 'Combinations with 5 or less observations')
    
    return df_few_obs
    
def prep_data_for_deepAR_univariate(df_sales_final, key_column, target_column, date_column):
    #DeepAR works in a set that use a index as date and columns as different time-series
    #key column distinguish each time series
    #It will have dates 
    #Multiple time series in DeepAR - Date as index and time series as columns
    print('Pivot table')
    df_sales_pivot = pd.pivot_table(data = df_sales_final, columns = 'key_Store_Dept',
                                    index = date_column, values = target_column)#.dropna(thresh=25, axis=1)
    df_sales_pivot = df_sales_pivot.fillna(0)
    df_sales_pivot[df_sales_pivot < 0] = 0
    print('There will be many values NAN in some keys. For those they will be replace for zeros')
    print('There is negative values in some Store-Dept sales combinations. We can keep as it is or change for zeros')
    return df_sales_pivot


def melt_df_back(df_univariate, split_date, date_column, key_column, target_column):
    data2 = pd.melt(df_univariate.reset_index(), id_vars=date_column, var_name=key_column, value_name=target_column)


    data2 = data2[['Date', 'key_Store_Dept', 'Weekly_Sales',]]
    train = data2.loc[data2['Date'] < split_date]
    valid = data2.loc[(data2['Date'] >= split_date) & (data2['Date'] < '2030-01-01')]    
    data2 = memory_aux.reduce_mem_usage(data2)
    train = memory_aux.reduce_mem_usage(train)
    valid = memory_aux.reduce_mem_usage(valid)
    
    return data2, train, valid

def define_date_index_split(df, split_ratio):
    end_date = df.index.max() 
    start_date = df.index.min() 
    date_interval = (end_date - start_date).days

    # Calculate the 90% point of the interval
    ninety_percent_point = start_date + timedelta(days=int(date_interval * split_ratio))
    
    print("Date interval:", date_interval, "days")
    print("90% Point:", ninety_percent_point.strftime("%Y-%m-%d"))
    split_date = pd.Timestamp(ninety_percent_point)  # Replace "yyyy-mm-dd" with your desired split date
    
    return split_date


def get_data(data_path, key_column, target_column, date_column, split_ratio):
    df = import_dataset(data_path)
    

    df_univariate = prep_data_for_deepAR_univariate(df, key_column, target_column, date_column)
    split_date = define_date_index_split(df_univariate, split_ratio)
    
    return df_univariate, split_date

def split_df(df,split_date ):
    train_data = df[df.index < split_date]  
    test_data = df[df.index >= split_date]  
    
    return train_data, test_data

def clean_plots_files(folder_path):
    files = os.listdir(folder_path)
    if len(files)>0:
        for file in files:
            os.remove(os.path.join(folder_path, f'{file}'))
    
    return 'Folder cleaned'
    
if __name__ == '__main__':
    
    print('teste')