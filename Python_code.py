########################################################################################################################################
#                                                     *** TIME SERIES FORECASTING ***                                                  #
#                                                                                                                                      #
# Author(s): Rishabh Kumar                                                                                                             #
# Creation Date: 20/02/2020                                                                                                            #
# Description: Store Item Demand Forecasting Challenge                                                                                 #
# Weblink: https://www.kaggle.com/c/demand-forecasting-kernels-only                                                                    #
#                                                                                                                                      #
#                                                             *** EDIT LOG ***                                                         #
# Last Edit Date: xx/xx/20xx                                                                                                           #
# Last Edit by: Rishabh Kumar                                                                                                          #
#                                                                                                                                      #
#        EDIT ID        |        EDITED BY         |          EDIT DESCRIPTION                                                         #  
#                                                                                                                                      #
########################################################################################################################################

########################################################################################################################################
#                                                             *** Steps Followed ***                                                   #
#                                                                                                                                      #
# Step 1: Import required libraries and read the Data                                                                                  #
# Step 2: Basic Data Exploration                                                                                                       #
# Step 3: Exploratory Data Analysis                                                                                                    #
# Step 4: Decomposition of Time Series and Seasonality Assessment (using ADF)                                                          #
# Step 5.a: Algorithm I - Moving Average                                                                                               #
# Step 5.b: Algorithm II - Exponential Smoothing                                                                                       #
# Step 5.c: Algorithm III -  ARIMA                                                                                                     #
# Step 5.c - (i): ACF And PACF plots                                                                                                   #
# Step 5.c - (ii): (S)ARIMA(X) - Feature Engineering + Model                                                                           #
# Step 5.d: XGBOOST                                                                                                                    #
# Step 5.e: Light Boost                                                                                                                #
# Step 5.f: FB Prophet                                                                                                                 #
# Step 5.g: Algorithm IV - Recurrent Neural Networks - LSTM                                                                            #
# Step 6: Model Comparison                                                                                                             #
# Step 7: Forecasting                                                                                                                  #
#                                                                                                                                      #
########################################################################################################################################



########################################################### START OF CODE ##############################################################

######################################################################################################################################## 
#                                                                                                                                      #
###################               STEP 1: IMPORT THE REQUIRED LIBRARIES AND READ THE DATA                  ############################# 
#                                                                                                                                      #
########################################################################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
# import datetime as datetime
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
# from matplotlib import pyplot
from matplotlib.pyplot import figure
from scipy.stats import skew
# import statsmodels.api as sm
from functools import reduce

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.holtwinters import HoltWintersResults
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.stats.outliers_influence import variance_inflation_factor

import pmdarima as pm

import xgboost as xgb
from xgboost import plot_importance, plot_tree

import lightgbm as lgb 
import datetime as dtime

# from fbprophet import Prophet

# Input and Output Paths
input_path = 'D:/Rishabh Kumar/Dropbox/2. Self Learning/1. Data Science and Analytics/2. Project(s)/1. Time Series Forecasting/1. Store Item Sales Forecasting/2. Raw Data/'
output_path = 'D:/Rishabh Kumar/Dropbox/2. Self Learning/1. Data Science and Analytics/2. Project(s)/1. Time Series Forecasting/1. Store Item Sales Forecasting/3. Output Data/'

# Read the data check
read_data = "YES"
# read_data = "NO"

# PLot Line Chart
# plot_line_charts = "YES"   # Will take a lot of time to run
plot_line_charts = "NO"

# PLot Histogram Charts
# plot_hist_charts = "YES"   # Will take a lot of time to run
plot_hist_charts = "NO"

# Analyse skewness check
# analyse_skewness = "YES"
analyse_skewness = "NO"

# Time Series Decompostion
# plot_decomp_charts = "YES"   # Will take a lot of time to run
plot_decomp_charts = "NO"

# Store and Item to focus
key_store_num_1 = 3
key_product_num_1 = 15

key_store_num_2 = 2
key_product_num_2 = 25

test_train_split = 0.75

# Simple Moving Average
sma_list = [[3,"green"],[5,"blue"],[7,"yellow"],[9,"black"],[14,"magenta"]]

# Simple Exponential Smoothing
ses_list = [.2,.4,.6]

# Holt's Linear Exponential Smoothing
holt_linear_list = [[.2,.001],[.2,.003],[.4,.001],[.4,.003],[.6,.001],[.6,.003]]

# Holt-Winter's Exponential Smoothing
# holt_winter_list = [['add','add',True],['add','add',False],['add','mul',True],['add','mul',False],['mul','mul',True],['mul','mul',False]]
holt_winter_list = [['add','mul',True],['mul','mul',True],['add','add',True]]
# holt_winter_list = [['add','add',True],['add','mul',True]]
period_num = 365

# Time Series Decomposition
period_var = 365    # Daily Data

# SARIMA
test_train_split_date = '2016-06-01'

#XGBOOST
time_delta_var = 150

# Output Forecast dataframe
output_forecast = "YES"
# output_forecast = "NO"

# Output Accuracy dataframe
output_accuracy = "YES"
# output_accuracy = "NO"


#############  READ THE DATA   ##############

if read_data == "YES":
    df = pd.read_csv(f"{input_path}train.csv", parse_dates = ['date'])
else:
    print("DATA NOT READ")
    
######################################################################################################################################## 
#                                                                                                                                      #   
############################                       STEP 2: BASIC DATA EXPLORATION                             ##########################
#                                                                                                                                      #
########################################################################################################################################

df.info() # Check
df.head() # Check
df.columns # Check
df.shape # Check
df.describe() # Check

# Counting 0 sales values
df_arry = df[df['sales'] == 0]['sales'].count()

# Create a backup of the main df
df_copy = df.copy()
# df_copy.head()

# Checking for null values: No Null values found
df.isnull().sum()

# Replace null values with mean or median
# df_copy.fillna(df_copy.mean(), inplace = True)      # Easier to explain and hence preferred method
# df_copy.fillna(df_copy.median(), inplace = True)    # Better than replacing with mean

######################################################################################################################################## 
#                                                                                                                                      #
#############################                       STEP 3: EXPLORATORY DATA ANALYSIS                     ##############################
#                                                                                                                                      #
########################################################################################################################################
### 3.1 Get unique values in the dataframe columns ###
        
# col_list = list(df_copy.select_dtypes(include='object'))
col_list = list(df_copy.columns)
print(col_list)

all_column_list = list(df.reindex(columns=col_list).apply(lambda x: list(x.unique()), axis=0))
print(len(all_column_list))

i = 0
for i in range (0,len(all_column_list)):
    print("Unique {} values are : {}".format(col_list[i],all_column_list[i]))
    i = i + 1

#### 3.2.1 Plotting Time Series in a line chart - via loop ###

# NOTE: The function below takes a lot of time to run 

def plot_line_chart_all():
    
    unique_store = list(df_copy.store.unique())
    unique_item = list(df_copy.item.unique())
    
    store_num = 1
    product_num = 1
    for store_num in unique_store:
        df_i = df_copy[df_copy['store'] == store_num]
        print(" ")
        # print("For Store {}".format(str(store_num)))
        for product_num in unique_item:
            df_j = df_i[df_i['item'] == product_num]
            print(" ")
            # print("For Item {}".format(str(product_num)))
            
            # Plotting the graph
            x = df_j['date'] 
            y = df_j['sales']
            figure(figsize=(10,6)) # Defining graph size
            plt.style.use('seaborn-notebook')
            plt.title("Sales for Store {} and Product/Item {} by Date".format(str(store_num),str(product_num))) # Graph Ttile
            plt.xlabel("Date") # X-axis label
            plt.ylabel("Sales") # Y-axis label
            plt.plot(x,y)
            plt.show()
            product_num = product_num + 1
        store_num=store_num+1

### 3.2.2 Plotting Time Series in a line chart - Sample plotting ###
def plot_line_chart_sample(store_num, product_num):

    date_filter = '2015-01-01'
    
    df_sample = df_copy[(df_copy['store'] == store_num) & (df_copy['item'] == product_num) & (df_copy['date'] > date_filter)]
    # df_sample.head() # Check
    # df_sample.shape # Check
    
    x_axis = df_sample['date']
    y_axis = df_sample['sales']
    
    figure(figsize = (10,6))
    plt.style.use('seaborn-notebook')
    plt.title("Sales for Store {} and Product/Item {} by Date".format(str(store_num),str(product_num)))
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.plot(x_axis,y_axis)
    plt.show()

       
if plot_line_charts == "YES":
    plot_line_chart_all()   #Takes a lot of time
else:
    print("ALL LINE CHARTS ARE NOT PLOTTED; ANALYSE THE SAMPLE LINE CHART")
    plot_line_chart_sample(store_num = key_store_num_1, product_num = key_product_num_1)
    plot_line_chart_sample(store_num = key_store_num_2, product_num = key_product_num_2)
    
# Conclusion: It appears that seasonality exists in the data 


### 3.3.1 Histogram plots of the data - via loop ###

# NOTE: The function below takes a lot of time to run 

def plot_histogram_all():
    
    unique_store = list(df_copy.store.unique())
    unique_item = list(df_copy.item.unique()) 
    
    var_name = "Sales"
    store_num = 1
    product_num = 1
    for store_num in unique_store:
        df_i = df_copy[df_copy['store'] == store_num]
        print(" ")
        # print("For Store {}".format(str(store_num)))
        for product_num in unique_item:
            df_j = df_i[df_i['item'] == product_num]
            print(" ")
            # print("For Item {}".format(str(product_num)))
            
            # Plotting the graph
            x_axis = df_j['sales']
            figure(figsize=(10,6)) # Defining graph size
            plt.style.use('seaborn-notebook')
            plt.title("{} frequency distribtuion for Store {} and Product/Item {}".format(var_name, str(store_num),str(product_num))) # Graph Ttile
            plt.xlabel("Value") # X-axis label
            plt.ylabel("Frequency") # Y-axis label
            plt.hist(x_axis,bins = None)
            plt.show()
            print("Skewness of the sales distribution for Store {} and Product/Item {} is: {}".format(store_num, product_num, skew(x_axis)))
            product_num = product_num + 1
        store_num=store_num+1

def plot_histogram_sample(store_num, product_num):
    
    var_name = 'sales'  
    
    df_sample = df_copy[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    
    x_hist = df_sample["{}".format(var_name)]
    
    figure(figsize = (10,6))
    plt.style.use('seaborn-notebook')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title("{} frequency distribution for Store {} and Product/Item {}".format(var_name.capitalize(), store_num, product_num))
    plt.hist(x_hist, bins = None)
    plt.show
    
    print("Skewness of the plot is: {}".format(skew(x_hist)))
       
if plot_hist_charts == "YES":
    plot_histogram_all()   #Takes a lot of time
    
else:
    print("ALL HISTOGRAMS ARE NOT PLOTTED; ANALYSE THE SAMPLE HISTOGRAM")
    plot_histogram_sample(store_num = key_store_num_1,product_num = key_product_num_1)
    plot_histogram_sample(store_num = key_store_num_1,product_num = key_product_num_2)
# Conclusion: It appears that the sales data is approximately normally distributed

### 3.3.2 Histogram plots of the data - Sample plotting ###
 


### 3.4.1 Summarising the data by stores ###

df_summarise = df_copy.copy()
print(df_summarise.columns)
df_summarise.info()
df_summarise = df_summarise.groupby(['date', 'item']).agg({'sales': sum}).reset_index()
df_copy.describe()
df_summarise.describe()

### 3.4.2 Plotting the summarised sales data ###

def summarize_plot():

    global skewness_dict
    var_name = 'sales'
    product_list = list(df_summarise['item'].unique())
    skewness_dict = {}
    
    i = 1 
    for i in range(1, max(product_list)+1):
        
        print("Plotting {} histogram for Product/Item {}".format(var_name, i))
        
        df_summarise_sample = df_summarise[(df_summarise['item'] == i)] 
        x_hist_summarize = df_summarise_sample["{}".format(var_name)]
        
        figure(figsize = (10,6))
        plt.style.use('seaborn-dark-palette')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title("{} frequency distribution for Product/Item {}".format(var_name.capitalize(), i))
        plt.hist(x_hist_summarize, bins = None)
        plt.show
        
        print("Skewness of the plot for Product/Item {} is: {}".format(i, skew(x_hist_summarize)))
        
        skewness_dict[i] = np.round(skew(x_hist_summarize),3)
        i = i+1

### 3.4.3 Plotting the skewness of the summarised sales data ###
        
def analyse_skewness_summarise():
    
    print("Range of skewness in the data is: ({} - {})".format(min(skewness_df['skewness']), max(skewness_df['skewness'])))
    
    skewness_df_sort = skewness_df.sort_values(['skewness']).reset_index(drop = True)
    # print(skewness_df_sort)
    
    colours = ['green']
    figure(figsize = (10,6))
    sns.set_style('darkgrid')
    x_axis = skewness_df_sort.index
    y_axis = skewness_df_sort['skewness']
    plt.title("Skewness by Items")    
    
    ax = sns.barplot( data = skewness_df_sort, x = x_axis, y = y_axis, palette = colours)
    ax.set_xticklabels(skewness_df_sort['item'])  
    
    print("The Sales data by item is approximately symmetric")

if analyse_skewness == "YES":
    
    # Function 1 Call
    summarize_plot() # Calling the function to print summarised plots
    
    print(skewness_dict)
    
    skewness_df = pd.DataFrame(skewness_dict.items(), columns = ['item', 'skewness'])
    skewness_df.head()
    
    # Function 2 Call
    analyse_skewness_summarise() # Calling the funtion to analyse skewness of the summarised plots
    
else:
    print("Skewness of the summarised data not analysed")
  
######################################################################################################################################## 
#                                                                                                                                      #    
################          STEP 4: DECOMPOSITION OF THE TIME SERIES AND SEASONALITY ASSESSMENT (USING ADF)              ################# 
#                                                                                                                                      #
########################################################################################################################################

### 4.1 Decompostion of the time series ###

# period_var = 7    # Daily Data

def time_series_decomp(df_name, decomp_type, period_num):
    
    unique_store = list(df_copy.store.unique())
    unique_item = list(df_copy.item.unique())
    
    store_num = 1
    product_num = 1
    for store_num in unique_store:
        df_i = df_name[df_copy['store'] == store_num]
        print(" ")
        # print("For Store {}".format(str(store_num)))
        for product_num in unique_item:
            df_j = df_i[df_i['item'] == product_num]
            df_time_decomp = df_j[['date','sales']].set_index('date')
            print(" ")
            # print("For Item {}".format(str(product_num)))
            
            # Plotting the graph
            figure(figsize=(10,6)) # Defining graph size
            plt.style.use('seaborn-notebook')
            print("{} Decomposition of sales for Store {} and Product/Item {}".format(decomp_type.upper(),str(store_num),str(product_num))) # Graph Ttile
            result = seasonal_decompose(df_time_decomp, model="{}".format(str(decomp_type)), period = period_num)
            result.plot()
            plt.show()
            product_num = product_num + 1
        store_num=store_num+1
        
def time_series_decomp_only1(decomp_type, period_num, store_num, product_num):
    print("ALL TIME SERIES ARE NOT DECOMPOSED; ANALYSE THE SAMPLE *{}* DECOMPOSITION OF TIME SERIES FOR STORE {} AND PRODUCT/ITEM {}".format(str(decomp_type).upper(),str(store_num),str(product_num)))
    
    df_time_decomp = df_copy[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    df_time_decomp = df_time_decomp[['date','sales']].set_index('date')
    
    figure(figsize=(10,6))
    plt.style.use('seaborn-notebook')
    result = seasonal_decompose(df_time_decomp, model="{}".format(str(decomp_type)), period = period_num)
    result.plot().suptitle(f"{decomp_type.title()} time series decomposition of sales for Store {store_num} and Product/Item {product_num} ", fontsize=13)
    # result.plot()
    plt.show()
    
### 4.1.1 Additive decompostion of the time series ###
        
# Additive Time Series: y(t) = Level + Trend + Seasonality + Noise
        
if plot_decomp_charts == "YES":
    time_series_decomp(df_name = df_copy,decomp_type = "additive", period_num = period_var)   #Takes a lot of time
else:    
    time_series_decomp_only1(decomp_type = "additive", period_num = period_var, store_num = key_store_num_1, product_num = key_product_num_1) 
    time_series_decomp_only1(decomp_type = "additive", period_num = period_var, store_num = key_store_num_2, product_num = key_product_num_2) 


### 4.1.2 Multiplicative decompostion of the time series ###

# Multiplicative Time Series: y(t) = Level * Trend * Seasonality * Noise

if plot_decomp_charts == "YES":
    
    # This step is done to remove 0 sales value from the data which causes an issue in multiplicative time series decomposition
    sales_array = df_copy[df_copy["sales"]!=0]["sales"]
    df_mul = df_copy.copy()
    df_mul["sales"] = df_mul["sales"].replace(0,sales_array.median())
    # df_mul.describe()
    
    time_series_decomp(df_name = df_mul, decomp_type = "multiplicative", period_num = period_var)   #Takes a lot of time
else:    
    time_series_decomp_only1(decomp_type = "multiplicative", period_num = period_var, store_num = key_store_num_1, product_num = key_product_num_1) 
    time_series_decomp_only1(decomp_type = "multiplicative", period_num = period_var, store_num = key_store_num_2, product_num = key_product_num_2) 

# Conclusion: Additive decomposition appears to better fit the data than multiplicative decomposition

### 4.1 Seasonality Assessment of the time series ###
    
######################################################################################################################################## 
# Link: https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/                                                #
#                                                                                                                                      #  
# IMPORTANT - Types of stationarity                                                                                                    #
#                                                                                                                                      #     
# 1. Strict Stationary: A strict stationary series satisfies the mathematical definition of a stationary process.                      #
# For a strict stationary series, the mean, variance and covariance are not the function of time.                                      #
# The aim is to convert a non-stationary series into a strict stationary series for making predictions.                                #
#                                                                                                                                      #
# 2. Trend Stationary: A series that has no unit root but exhibits a trend is referred to as a trend stationary series.                #
# Once the trend is removed, the resulting series will be strict stationary.                                                           #
# The KPSS test classifies a series as stationary on the absence of unit root.                                                         #
# This means that the series can be strict stationary or trend stationary.                                                             #
#                                                                                                                                      #    
# 3. Difference Stationary: A time series that can be made strict stationary by differencing falls under difference stationary.        #
# ADF test is also known as a difference stationarity test.                                                                            #
######################################################################################################################################## 
                                                                                                                               
######################################################################################################################################## 
# IMPORTANT - How to interpret ADF and KPSS results                                                                                   #    
#                                                                                                                                      # 
# Case 1: Both tests conclude that the series is not stationary -> series is not stationary                                            #
# Case 2: Both tests conclude that the series is stationary -> series is stationary                                                    #
# Case 3: KPSS = stationary and ADF = not stationary  -> trend stationary, remove the trend to make series strict stationary           #
# Case 4: KPSS = not stationary and ADF = stationary -> difference stationary, use differencing to make series stationary              #
########################################################################################################################################
    
### KPSS - To Identify Trend stationarity ###
    
# Null hypothesis: The process is trend stationary
# Alternate hypothesis: The series has a unit root (series is not stationary)

# period_var = 7
# period_var = 365

def kpss_test(store_num, product_num, season_var, n_periods, diff):
    print("\nResult of KPSS test")
    df_sta_test = df_copy[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)]  
    if season_var == "REMOVE SEASONALITY" and diff == "DIFFERENCE":
        df_sta_test['sales_wo_season'] = df_sta_test['sales'].diff(n_periods)
        kpsstest = kpss(df_sta_test['sales_wo_season'].diff().dropna(), regression='c')
    elif season_var == "REMOVE SEASONALITY" and diff != "DIFFERENCE":
        df_sta_test['sales_wo_season'] = df_sta_test['sales'].diff(n_periods)
        kpsstest = kpss(df_sta_test['sales_wo_season'].dropna(), regression='c')
    else:
        kpsstest = kpss(df_sta_test['sales'], regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(f'{"Seasonality removded" if season_var == "REMOVE SEASONALITY" else "Seasonlity not removed"} and {"Differencing done" if diff == "DIFFERENCE" else "Differencing not done"}\n')
    print(kpss_output)
    print(f'Result (KPSS): The series for Store {store_num} and Product/Item {product_num} is {"*NOT* " if kpss_output["p-value"] < 0.05 else ""}stationary \n')

# No Seasonality and No Differencing done
kpss_test(store_num = key_store_num_1, product_num = key_product_num_1, season_var = "DO NOT REMOVE SEASONALITY", n_periods = period_var, diff = "NO DIFFERENCE")    
kpss_test(store_num = key_store_num_2, product_num = key_product_num_2, season_var = "DO NOT REMOVE SEASONALITY", n_periods = period_var, diff = "NO DIFFERENCE")

# Seasonality Removed but no Differencing done
kpss_test(store_num = key_store_num_1, product_num = key_product_num_1, season_var = "REMOVE SEASONALITY", n_periods = period_var, diff = "NO DIFFERENCE")    
kpss_test(store_num = key_store_num_2, product_num = key_product_num_2, season_var = "REMOVE SEASONALITY", n_periods = period_var, diff = "NO DIFFERENCE")

# Seasonality and Differencing done
kpss_test(store_num = key_store_num_1, product_num = key_product_num_1, season_var = "REMOVE SEASONALITY", n_periods = period_var, diff = "DIFFERENCE")    
kpss_test(store_num = key_store_num_2, product_num = key_product_num_2, season_var = "REMOVE SEASONALITY", n_periods = period_var, diff = "DIFFERENCE")

### ADF - To Identify Difference stationarity ###
    
# Null hypothesis: The series has a unit root (value of a = 1)
# Alternate hypothesis: The series has no unit root

def adf_test(store_num, product_num):
    print("\nResult of ADF test")
    df_sta_test = df_copy[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)]  
    adftest = adfuller(df_sta_test['sales'], autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    print(adfoutput)
    print(f'Result (ADF): The series for Store {store_num} and Product/Item {product_num} is {"" if adfoutput["p-value"] < 0.05 else "*NOT* "}stationary \n')
    
adf_test(store_num = key_store_num_1, product_num = key_product_num_1)    
adf_test(store_num = key_store_num_2, product_num = key_product_num_2)

# Conclusion: We need to difference the time series to make it stationary
######################################################################################################################################## 
#                                                                                                                                      #    
#########################                      STEP 5: TIME SERIES FORECASTING ALGORITHMS                   ############################ 
#                                                                                                                                      #
########################################################################################################################################

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

######################################################################################################################################## 
#                                                                                                                                      #    
#########################                                    MOVING AVERAGE                                 ############################ 
#                                                                                                                                      #
########################################################################################################################################

# IMPORTANT: Eval function is a security red flag

# sma_list = [[3,"green"],[5,"blue"],[7,"yellow"],[9,"black"],[14,"magenta"]]

def simple_moving_average(store_num, product_num, list_name, plot_start):

    df_moving_avg = df_copy.copy()
    df_moving_avg = df_moving_avg[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    
    i = 0
    while i <len(list_name):
        df_moving_avg[f'rolling_mean_{list_name[i][0]}'] = df_moving_avg.rolling(window=list_name[i][0], on='date').sales.mean()
        i = i+1
        
    # Plot in a grpah     
    figure(figsize=(10,6))  
    plt.style.use('seaborn-dark-palette')
    x_axis = df_moving_avg['date'][plot_start:]
    y_axis = df_moving_avg['sales'][plot_start:]
    
    plt.plot(x_axis, y_axis, label='Sales')
    
    i = 0
    while i <len(list_name):
        plt.plot(x_axis, df_moving_avg[f"rolling_mean_{list_name[i][0]}"][plot_start:], label= f'Sales {list_name[i][0]} Day SMA', color= f'{list_name[i][1]}')
        # plt.plot(x_axis, "df_moving_avg.rolling_mean_{}".format(list_name[i][0]), label= "Sales {} Day SMA".format(list_name[i][0]), color= "{}".format(list_name[i][1]))
        i = i+1
         
    plt.title(f'Simple Moving Average Sales by for Store {store_num} and Product/Item {product_num}')
    plt.legend(loc='upper left')
    plt.show() 
    
    # Compute MAPE, MSE, RMSE
    i = 0
    print(f"\nFor Store {store_num} and Product/Item {product_num}:")
    while i <len(list_name):      
        
        y_true = df_moving_avg['sales']
        y_pred = df_moving_avg[f"rolling_mean_{list_name[i][0]}"]
        y_true = np.array(y_true[plot_start:])
        y_pred = np.array(y_pred[plot_start:])     
        print(f"MAPE for window_size {list_name[i][0]} is: {round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)}")
        print(f"MSE for window_size {list_name[i][0]} is: {round(np.mean((np.square(y_pred - y_true))),2)}")
        print(f"RMSE for window_size {list_name[i][0]} is: {round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)} \n")
        i = i+1
    
simple_moving_average(store_num = key_store_num_1, product_num = key_product_num_1, list_name = sma_list, plot_start = -500)    
simple_moving_average(store_num = key_store_num_2, product_num = key_product_num_2, list_name = sma_list, plot_start = -500)

######################################################################################################################################## 
#                                                                                                                                      #    
#########################                                 EXPONENTIAL SMOOTHING                             ############################ 
#                                                                                                                                      #
########################################################################################################################################

### 5.b.1 Single Exponential Smoothing ###

# ses_list = [.2,.4,.6]
# store_num=2
# product_num = 25
# list_name = [.2,.4,.6]
# split_date = '2016-06-30'
    
def simple_exp_smt(store_num, product_num, list_name, split_date):
    

    df_exp_smo= df_copy.copy()
    df_exp_smo = df_exp_smo[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    
    # num_rows = round(test_train*len(df_exp_smo['sales']))
    
    # train = df_exp_smo[:num_rows]
    # test = df_exp_smo[num_rows+1:]
    train = df_exp_smo[(df_exp_smo['date'] < f'{split_date}')]
    test = df_exp_smo[(df_exp_smo['date'] >= f'{split_date}')]
    
    train.index = pd.to_datetime(train['date'])
    test.index = pd.to_datetime(test['date'])
    test = test.copy()
    
    i = 0
    while i <len(list_name):        
        fit = SimpleExpSmoothing(np.asarray(train['sales'])).fit(smoothing_level=list_name[i],optimized=False)
        test[f'SES_{round(list_name[i]*10)}'] = fit.forecast(len(test))
        i = i+1
    
    figure(figsize=(10,6))
    plt.style.use('seaborn-dark-palette')
    plt.plot(train['sales'][-500:], label='Train')
    plt.plot(test['sales'], label='Test')
    i = 0
    while i <len(list_name):
        plt.plot(test[f'SES_{round(list_name[i]*10)}'], label=f'alpha = {list_name[i]}')
        i = i +1
    plt.title(f"Simple Exponential Smoothing for Store {store_num} and Product/Item {product_num}")  
    plt.legend(loc='upper left')
    plt.show()
    # test.info()
    
    i = 0
    print(f"\nFor Store {store_num} and Product/Item {product_num}:")
    while i <len(list_name):      
        
        y_true = test['sales']
        y_pred = test[f"SES_{round(list_name[i]*10)}"]
        y_true = np.array(y_true[:])
        y_pred = np.array(y_pred[:])     
        print(f"MAPE for alpha = {list_name[i]} is: {round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)}")
        print(f"MSE for alpha = {list_name[i]} is: {round(np.mean((np.square(y_pred - y_true))),2)}")
        print(f"RMSE for alpha = {list_name[i]} is: {round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)} \n")
        i = i+1
        
simple_exp_smt(store_num = key_store_num_1, product_num = key_product_num_1, list_name = ses_list, split_date = test_train_split_date)    
simple_exp_smt(store_num = key_store_num_2, product_num = key_product_num_2, list_name = ses_list, split_date = test_train_split_date)

######################################################################################################################################## 
#                                                                                                                                      #    
#########################                                    HOLT'S LINEAR                                  ############################ 
#                                                                                                                                      #
########################################################################################################################################

### 5.b.2 Double Exponential Smoothing (includes Trend) ###

# holt_linear_list = [[.2,.001],[.2,.003],[.4,.001],[.4,.003],[.6,.001],[.6,.003]]

def holt_linear_exp_smt(store_num, product_num, list_name, split_date):

    
    # store_num=2
    # product_num = 25
    # list_name = [[.2,.001],[.2,.003],[.4,.001],[.4,.003],[.6,.001],[.6,.003]]
    # split_date = '2016-06-30'
    
    df_exp_smo= df_copy.copy()
    df_exp_smo = df_exp_smo[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    
    # num_rows = round(test_train*len(df_exp_smo['sales']))
    
    # train = df_exp_smo[:num_rows]
    # test = df_exp_smo[num_rows+1:]
    train = df_exp_smo[(df_exp_smo['date'] < f'{split_date}')]
    test = df_exp_smo[(df_exp_smo['date'] >= f'{split_date}')]
    test = test.copy()

    i = 0
    while i <len(list_name):
        fit = Holt(np.asarray(train['sales'])).fit(smoothing_level = list_name[i][0],smoothing_slope = list_name[i][1])
        test[f'Holt_linear_{round(list_name[i][0]*10)}_{list_name[i][1]}'] = fit.forecast(len(test))
        i = i +1
    
    # plt.figure(figsize=(16,8))
    train.index = pd.to_datetime(train['date'])
    test.index = pd.to_datetime(test['date'])
         
    figure(figsize=(10,6))  
    plt.style.use('seaborn-dark-palette')
    plt.plot(train['sales'][-500:], label='Train')
    plt.plot(test['sales'], label='Test')
    
    i = 0
    while i <len(list_name):
        plt.plot(test[f'Holt_linear_{round(list_name[i][0]*10)}_{list_name[i][1]}'], label=f'Holt_linear_{round(list_name[i][0]*10)}_{list_name[i][1]}')
        i = i +1
        
    plt.title(f"Holt's Linear Smoothing for Store {store_num} and Product/Item {product_num}")      
    plt.legend(loc='upper left')
    plt.show()
    
        
    i = 0
    print(f"\nFor Store {store_num} and Product/Item {product_num}:")
    while i <len(list_name):      
        
        y_true = test['sales']
        y_pred = test[f'Holt_linear_{round(list_name[i][0]*10)}_{list_name[i][1]}']
        y_true = np.array(y_true[:])
        y_pred = np.array(y_pred[:])     
        print(f"MAPE for alpha = {list_name[i][0]} and trend = {list_name[i][1]} is: {round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)}")
        print(f"MSE for alpha = {list_name[i][0]} and trend = {list_name[i][1]} is: {round(np.mean((np.square(y_pred - y_true))),2)}")
        print(f"RMSE for alpha = {list_name[i][0]} and trend = {list_name[i][1]} is: {round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)} \n")
        i = i+1
    
holt_linear_exp_smt(store_num = key_store_num_1, product_num = key_product_num_1, list_name = holt_linear_list, split_date = test_train_split_date)  
holt_linear_exp_smt(store_num = key_store_num_2, product_num = key_product_num_2, list_name = holt_linear_list, split_date = test_train_split_date)  

######################################################################################################################################## 
#                                                                                                                                      #    
#########################                                    HOLT'S WINTER                                  ############################ 
#                                                                                                                                      #
########################################################################################################################################

### 5.b.3 Tripe Exponential Smoothing (includes Trend and Seasonality) ###

# holt_winter_list = [['add','add',True],['add','add',False],['add','mul',True],['add','mul',False],['mul','mul',True],['mul','mul',False]]
# holt_winter_list = [['add','add',True],['add','add',False],['add','mul',True],['add','mul',False]]
# holt_winter_list = [['add','add',True],['add','mul',True]]

def holt_winter_exp_smt(store_num, product_num, list_name, period, split_date):
    
    # store_num = key_store_num_1
    # product_num = key_product_num_1
    list_name = holt_winter_list
    period = period_num
    # split_date = test_train_split_date
    
    df_exp_smo= df_copy.copy()
    df_exp_smo = df_exp_smo[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    df_exp_smo.index = df_exp_smo['date']
    
    # num_rows = round(test_train*len(df_exp_smo['sales']))
    
    # train = df_exp_smo[:num_rows]
    # test = df_exp_smo[num_rows+1:]
    train = df_exp_smo[(df_exp_smo['date'] < f'{split_date}')]
    test = df_exp_smo[(df_exp_smo['date'] >= f'{split_date}')]
    # len(train)
    # len(test)
    
    i = 0
    df_holt_winter_param = None
    while i < len(list_name):
        fit = ExponentialSmoothing(np.asarray(train['sales']) , seasonal_periods=period ,trend=list_name[i][0], seasonal=list_name[i][1],damped = list_name[i][2]).fit()
        test[f'Holt_Winter_{list_name[i][0]}_{list_name[i][1]}_{list_name[i][2]}'] = fit.forecast(len(test))
        
        df_check = fit.params_formatted.iloc[:6,:]
        df_check['trend'] = list_name[i][0]
        df_check['seasonal'] = list_name[i][1]
        df_check['damped'] = list_name[i][2]
    
        if df_holt_winter_param is None:
            df_holt_winter_param = df_check
        else:
            df_holt_winter_param = df_holt_winter_param.append(df_check)
            
        # print(HoltWintersResults(fit) #Check
        i = i+1
    
    # plt.figure(figsize=(10,6))         
    figure(figsize=(10,6))  
    plt.style.use('seaborn-dark-palette')
    plt.plot(train['sales'][-200:], label='Train')
    plt.plot(test['sales'], label='Test')
    i = 0
    while i < len(list_name):
        plt.plot(test[f'Holt_Winter_{list_name[i][0]}_{list_name[i][1]}_{list_name[i][2]}'], label=f'Holt_Winter_{list_name[i][0]}_{list_name[i][1]}_{list_name[i][2]}')
        i = i +1

    plt.title(f"Holt-Winter's Smoothing for Store {store_num} and Product/Item {product_num}")  
    plt.legend(loc='upper left')
    plt.show()
    
    i = 0
    df_accuracy_hw = None
    print(f"\nFor Store {store_num} and Product/Item {product_num}:")
    while i <len(list_name):      
        
        y_true = test['sales']
        y_pred = test[f'Holt_Winter_{list_name[i][0]}_{list_name[i][1]}_{list_name[i][2]}']
        y_true = np.array(y_true[:])
        y_pred = np.array(y_pred[:])    
        
        MAPE_var = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)
        MSE_var = round(np.mean((np.square(y_pred - y_true))),2)
        RMSE_var = round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)
        
        print(f"MAPE for trend = {list_name[i][0]},  seasonal_trend = {list_name[i][1]} and damping = {list_name[i][2]} is: {MAPE_var}")
        print(f"MSE for alpha = {list_name[i][0]},  seasonal_trend = {list_name[i][1]} and damping = {list_name[i][2]} is: {MSE_var}")
        print(f"RMSE for alpha = {list_name[i][0]},  seasonal_trend = {list_name[i][1]} and damping = {list_name[i][2]} is: {RMSE_var} \n")
        
        df_check = pd.DataFrame({'ID': [f'Holt_Winter_{list_name[i][0]}_{list_name[i][1]}_{list_name[i][2]}'], 'Store': [f'{store_num}'], 'Product':[f'{product_num}'],
                                    'MAPE':[f'{MAPE_var}'],'MSE':[f'{MSE_var}'],'RMSE':[f'{RMSE_var}']}) 
        if df_accuracy_hw is not None:
            df_accuracy_hw = pd.concat([df_accuracy_hw, df_check])
        else:
            df_accuracy_hw = df_check.copy()
        i = i+1
        
    # df_test_forecast = test.copy().drop(columns = ['date','sales'])    
    df_test_forecast = test.copy().drop(columns = ['date'])        
    return df_test_forecast, df_accuracy_hw, df_holt_winter_param
        
df_forecast_1, df_accuracy_1, df_holt_winter_param1 = holt_winter_exp_smt(store_num = key_store_num_1, product_num = key_product_num_1, list_name = holt_winter_list, period = period_num, split_date = test_train_split_date)  
df_forecast_2, df_accuracy_2, df_holt_winter_param2 = holt_winter_exp_smt(store_num = key_store_num_2, product_num = key_product_num_2, list_name = holt_winter_list, period = period_num, split_date = test_train_split_date)  

df_holt_winter_forecast = pd.concat([df_forecast_1,df_forecast_2])
df_holt_winter_accuracy = pd.concat([df_accuracy_1,df_accuracy_2])
df_holt_winter_param = pd.concat([df_holt_winter_param1,df_holt_winter_param2])

df_holt_winter_forecast.head()
df_holt_winter_accuracy.head(20)
df_holt_winter_param.head(10)

########################################################################################################################################
#                                                                                                                                      #
#####################################                    ACF AND PACF PLOTS                        #####################################
#                                                                                                                                      #
########################################################################################################################################
                                                                            

### 5.c.1 ACF and PACF Plots ###

# store_num = key_store_num_1
# product_num = key_product_num_1
# test_train = 0.75
# n_lags = 75
# n_periods_week = 7
# n_periods_annual = 365

def acf_pacf_plots(store_num, product_num, test_train, n_lags, n_periods_annual):

    df_arima = df_copy.copy()
    df_arima = df_arima[(df_copy['store'] == store_num) & (df_copy['item'] == product_num)] 
    df_arima.index = df_arima['date']
    
    # Visualising the seasonality in the data
    n_points = 365
    plt.plot(df_arima['sales'][-n_points:])
    plt.title(f"Sales for last {n_points} days")
    plt.show()
    
    # No Seasonality removed and no differencing
    plot_acf(df_arima['sales'], ax=plt.gca(), lags = n_lags, title = "Normal ACF Plot")
    plt.show()
    plot_pacf(df_arima['sales'], method = 'ywm', ax=plt.gca(), lags = n_lags, title = "Normal PACF Plot")
    plt.show()
    
    # Remove seasonality    
    # df_arima['sales_wo_season'] = df_arima['sales'].diff(n_periods_week)
    df_arima['sales_wo_season']= df_arima['sales'].diff(n_periods_annual)
    # df_arima.tail(500)
    
    plot_acf(df_arima['sales_wo_season'].dropna(), ax=plt.gca(), lags = n_lags, title = f"ACF Plot after removing Seasonaility(n_periods = {n_periods_annual})")
    plt.show()
    plot_pacf(df_arima['sales_wo_season'].dropna(), ax=plt.gca(), method = 'ywm', lags = n_lags,title = f"PACF Plot after removing Seasonaility(n_periods = {n_periods_annual})")
    plt.show()
    
    # Remove seasonality and differencing done
    plot_acf(df_arima['sales_wo_season'].diff().dropna(), ax=plt.gca(), lags = n_lags, title = f"ACF Plot after removing Seasonaility(n_periods = {n_periods_annual})\n and differencing")
    plt.show()
    plot_pacf(df_arima['sales_wo_season'].diff().dropna(), ax=plt.gca(), method = 'ywm', lags = n_lags, title = f"PACF Plot after removing Seasonaility(n_periods = {n_periods_annual})\n and differencing")
    plt.show()

acf_pacf_plots(store_num = key_store_num_1, product_num = key_product_num_1, test_train = test_train_split, n_lags = 75, n_periods_annual =7)
acf_pacf_plots(store_num = key_store_num_2, product_num = key_product_num_2, test_train = test_train_split, n_lags = 75, n_periods_annual =7)

########################################################################################################################################
#                                                                                                                                      #
#####################################                        AUTO ARIMA                        #########################################
#                                                                                                                                      #
########################################################################################################################################

### 5.c.3 ARIMA ###
# Good to read: https://alkaline-ml.com/pmdarima/tips_and_tricks.html#period

# test_train_split_date = '2016-06-01'

def auto_arima_without_exo(store_num, product_num, split_date):
    # store_num = key_store_num_1
    # product_num = key_product_num_1
    # test_train = 0.75
    # split_date = '2016-06-01'
    
    df_arima = df_copy.copy()
    df_arima = df_arima[(df_arima['store'] == store_num) & (df_arima['item'] == product_num)] 
    df_arima.index = df_arima['date']
    
    # num_rows = round(test_train*len(df_arima['sales']))
    
    train = df_arima[(df_arima['date'] < f'{split_date}')]
    test = df_arima[(df_arima['date'] >= f'{split_date}')]
    
    # Estimate ARIMA differencing term
    diff_term = pm.arima.ndiffs(train['sales'], max_d = 3, test='kpss')
    
    # Estimate ARIMA seasonal differencing
    season_diff_term = pm.arima.nsdiffs(train['sales'], max_D = 3, m=7, test='ocsb')
    
    model = pm.auto_arima(train['sales'],
                          start_p=1, start_q=1,
                          max_p = 7, max_q = 7,
                          start_P = 1, start_Q = 1,
                          max_P = 5, max_Q = 5,
                          max_order = None,
                          trace=True,
                          d=diff_term,
                          test = 'kpss',
                          seasonal=True,
                          D=season_diff_term,
                          # m=365,
                          m=7,
                          information_criterion='aicc',
                          error_action='ignore',
                          suppress_warnings=True,
                          random_state = 123,
                          stepwise=True)
    
    print(f'AUTO ARIMA (*without* exogenous variables) results for Store {store_num} and Product/Item {product_num}')    
    
    results = model.fit(train['sales'])
    print(results.summary())
    
    normal_order = results.get_params().get("order")
    seasonal_order = results.get_params().get("seasonal_order")
    
    new_dict = {"Store" : f'{store_num}', "Product" : f'{product_num}', "Order" : f'{normal_order}',
                "Seasonal_order" : f'{seasonal_order}', "Differencing" : f'{diff_term}',
                "Seasonal_differencing" : f'{season_diff_term}'}
    
    results.plot_diagnostics()
    plt.show()
    
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
    forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
    conf_int = pd.DataFrame(conf_int,index = test.index, columns=['Pred_LCI','Pred_UCI'])
    conf_int.head()
    forecast.head()
    forecast = forecast.merge(conf_int, how = 'inner', left_index = True, right_index = True)
    
    #plot the predictions for validation set
    figure(figsize=(10,6))
    plt.style.use('seaborn-dark-palette')
    plt.plot(train['sales'][-100:], label='Train')
    plt.plot(test['sales'], label='Test')
    plt.plot(forecast['Prediction'], label='Prediction')
    
    # plt.plot(forecast['Pred_LCI'], 'g--', label='Pred_LCI')
    # plt.plot(forecast['Pred_UCI'], 'y--', label='Pred_UCI')    
    plt.fill_between(forecast.index, forecast['Pred_LCI'],  forecast['Pred_UCI'], alpha=0.3, color='g', label='Confidence Interval')
    
    plt.title(f'AUTO ARIMA (*without* exogenous variables) - Actual vs Forecast for Store {store_num} and Product/Item {product_num}') 
    plt.legend(loc='upper left')
    plt.show()
    
    df_test_forecast = forecast.copy().rename(columns={"Prediction":"Prediction_arima", "Pred_LCI":"Pred_arima_LCI",
                                                                "Pred_UCI":"Pred_arima_UCI"})
    df_test_forecast = df_test_forecast.merge(test[['store','item','sales']], how='inner', left_index= True,
                                              right_index = True)
 
    # Checking accuracy of the model
    y_true = test['sales']
    y_pred = forecast['Prediction']
    y_true = np.array(y_true[:])
    y_pred = np.array(y_pred[:]) 
    
    MAPE_var = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)
    MSE_var = round(np.mean((np.square(y_pred - y_true))),2)
    RMSE_var = round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)
    
    print(f"MAPE for Store {store_num} and Product/Item {product_num} is: {MAPE_var}")
    print(f"MSE for Store {store_num} and Product/Item {product_num} is: {MSE_var}")
    print(f"RMSE for Store {store_num} and Product/Item {product_num} is: {RMSE_var} \n")
    
    df_accuracy = pd.DataFrame({'ID': ['Auto_SARIMAX_no_exo'], 'Store': [f'{store_num}'], 'Product':[f'{product_num}'],
                            'MAPE':[f'{MAPE_var}'],'MSE':[f'{MSE_var}'],'RMSE':[f'{RMSE_var}']}) 
    # return df_test_forecast, df_accuracy
    return df_test_forecast, df_accuracy, new_dict

df_forecast_1, df_accuracy_1, autoarima_no_exo_param_dict_1= auto_arima_without_exo(store_num = key_store_num_1, product_num = key_product_num_1, split_date = test_train_split_date)
df_forecast_2, df_accuracy_2, autoarima_no_exo_param_dict_2= auto_arima_without_exo(store_num = key_store_num_2, product_num = key_product_num_2, split_date = test_train_split_date)

df_autoarima_no_exo_forecast = pd.concat([df_forecast_1,df_forecast_2])
df_autoarima_no_exo_accuracy = pd.concat([df_accuracy_1,df_accuracy_2])

########################################################################################################################################
#                                                                                                                                      #
###################################             AUTO ARIMA WITH EXOGENOUS VARIABLES            #########################################
#                                                                                                                                      #
########################################################################################################################################

### Add exogenous variables on month, day of the week, week_number, date of the month ###

# Creating exogenous variables for the complete dataset

df_arima_exo = df_copy.copy()
# df_arima_exo = df_arima_exo[(df_arima_exo['store'] == store_num) & (df_arima_exo['item'] == product_num)] 
df_arima_exo.index = df_arima_exo['date']

# df_arima_exo['year'] = df_arima_exo['date'].dt.year
# df_arima_exo['quarter'] = df_arima_exo['date'].dt.quarter
df_arima_exo['month'] = df_arima_exo['date'].dt.month
# df_arima_exo['day'] = df_arima_exo['date'].dt.day

# df_arima_exo['week_year'] = df_arima_exo['date'].dt.weekofyear
df_arima_exo['day_week_num'] = df_arima_exo['date'].dt.weekday

# Creating Weekday - Weekend variable
df_arima_exo['weekday_id'] = np.where(df_arima_exo['day_week_num'] < 5, 1, 0) # 1- Weekday, 0 - Weekend

rolling_win = 7
df_arima_exo[f'rolling_mean_{rolling_win}'] = df_arima_exo.rolling(window=rolling_win, on='date')['sales'].mean()

# df_arima_exo.info()
# df_arima_exo.head()

# print(df_arima_exo['week_year'])
print(df_arima_exo['day_week_num'])

# Performing Dummy encoding

# features_to_encode = ['month','week_year','day', 'day_week_num']
features_to_encode = ['month','day_week_num']

def encode_and_bind(original_dataframe, feature_to_encode):
    df_arima_exo_update = original_dataframe.copy()
    i = 0
    while i < len(features_to_encode):
        dummies = pd.get_dummies(df_arima_exo_update[f'{feature_to_encode[i]}'], prefix = f'{feature_to_encode[i]}',drop_first=True)
        df_arima_exo_update = pd.concat([df_arima_exo_update, dummies], axis=1)    
        df_arima_exo_update = df_arima_exo_update.drop([f'{feature_to_encode[i]}'], axis=1)
        i = i + 1
    return(df_arima_exo_update)

# for feature in features_to_encode:
#     df_arima_exo_update = encode_and_bind(original_dataframe = df_arima_exo, feature_to_encode = feature)
df_arima_exo_update = encode_and_bind(original_dataframe = df_arima_exo, feature_to_encode = features_to_encode)
    
# df_arima_exo.info()
# df_arima_exo_update.info()
# df_arima_exo_update.head()
# df_arima_exo_update.to_excel("check.xlsx")

# Multi collinearity check

def auto_arima_with_exo(store_num, product_num, split_date):

    df_auto_arima_new = df_arima_exo_update.copy()    
    df_auto_arima_new = df_auto_arima_new[(df_auto_arima_new['store'] == store_num) & (df_auto_arima_new['item'] == product_num)] 
    
    df_arima_multi_col_check = df_auto_arima_new.copy()
    df_arima_multi_col_check = df_arima_multi_col_check.drop(columns = ['date', 'store', 'item', 'sales'])
    df_arima_multi_col_check.info()

    # NOT USING VIF since dummy variables have been created
    # Removing the collinear variables using VIF 
    
    # def calculate_vif_(X, thresh):
    #     variables = list(range(X.shape[1]))
    #     dropped = True
    #     while dropped:
    #         dropped = False
    #         vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
    
    #         maxloc = vif.index(max(vif))
    #         if max(vif) > thresh:
    #             print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
    #                   '\' at index: ' + str(maxloc))
    #             del variables[maxloc]
    #             dropped = True
    
    #     print('Remaining variables:')
    #     print(X.columns[variables])
    #     return X.iloc[:, variables]
    
    # keep_cols = list(calculate_vif_(X = df_arima_multi_col_check, thresh = 5.0))
    # print(keep_cols)
    
    # keep_cols.extend(['date', 'store', 'item', 'sales'])
    # print(keep_cols)
    
    # df_arima_multi_col_check_2 = df_auto_arima_new.copy()
    # df_arima_multi_col_check_2 = df_arima_multi_col_check_2[keep_cols]
    # df_arima_multi_col_check_2.info()
    
    df_arima_multi_col_check_2 = df_auto_arima_new.copy()
    
    # 
    train_with_exo = df_arima_multi_col_check_2[(df_arima_multi_col_check_2['date'] < f'{split_date}')]
    test_with_exo = df_arima_multi_col_check_2[(df_arima_multi_col_check_2['date'] >= f'{split_date}')]
    
    # Estimate ARIMA differencing term
    diff_term = pm.arima.ndiffs(train_with_exo['sales'], max_d = 3, test='kpss')
    
    # Estimate ARIMA seasonal differencing
    season_diff_term = pm.arima.nsdiffs(train_with_exo['sales'], max_D = 3, m=7, test='ocsb')
    
    train_only_exo = train_with_exo.copy()
    train_only_exo = train_only_exo.drop(columns = ['date', 'sales', 'store', 'item','weekday_id'])
    
    test_only_exo = test_with_exo.copy()
    test_only_exo = test_only_exo.drop(columns = ['date', 'sales', 'store', 'item','weekday_id'])
    
    
    ### Running Aut ARIMA code ###
    
    model_exco = pm.auto_arima(train_with_exo['sales'],
                          exogenous = train_only_exo,
                          start_p=1, start_q=1,
                          max_p = 7, max_q = 7,
                          start_P = 1, start_Q = 1,
                          max_P = 5, max_Q = 5,
                          max_order = None,
                          trace=True,
                          d=diff_term,
                          test = 'kpss',
                          seasonal=True,
                          D=season_diff_term,
                          # m=365,
                          m=7,
                          information_criterion='aicc',
                          error_action='ignore',
                          suppress_warnings=True,
                          random_state = 123,
                          stepwise=True)
    
    print(f'AUTO ARIMA (*with* exogenous variables) results for Store {store_num} and Product/Item {product_num}')
        
    results_exco = model_exco.fit(train_with_exo['sales'], exogenous = train_only_exo)
    print(results_exco.summary())
    
    results_exco.plot_diagnostics()
    plt.show()
        
    normal_order = results_exco.get_params().get("order")
    seasonal_order = results_exco.get_params().get("seasonal_order")
    
    new_dict = {"Store" : f'{store_num}', "Product" : f'{product_num}', "Order" : f'{normal_order}',
                "Seasonal_order" : f'{seasonal_order}', "Differencing" : f'{diff_term}',
                "Seasonal_differencing" : f'{season_diff_term}'}
    
    forecast_with_exo, conf_int = model_exco.predict(n_periods=len(test_with_exo), exogenous = test_only_exo, return_conf_int=True)
    forecast_with_exo = pd.DataFrame(forecast_with_exo,index = test_with_exo.index,columns=['Prediction'])
    conf_int = pd.DataFrame(conf_int,index = test_with_exo.index, columns=['Pred_LCI','Pred_UCI'])
    conf_int.head()
    forecast_with_exo.head()
    forecast_with_exo = forecast_with_exo.merge(conf_int, how = 'inner', left_index = True,
                                                right_index = True)
    
    #plot the predictions for validation set
    figure(figsize=(10,6))
    plt.style.use('seaborn-dark-palette')
    plt.plot(train_with_exo['sales'][-100:], label='Train')
    plt.plot(test_with_exo['sales'], label='Test')
    plt.plot(forecast_with_exo['Prediction'], label='Prediction')
    
    
    # plt.plot(forecast_with_exo['Pred_LCI'], 'g--', label='Pred_LCI')
    # plt.plot(forecast_with_exo['Pred_UCI'], 'y--', label='Pred_UCI')    
    plt.fill_between(forecast_with_exo.index, forecast_with_exo['Pred_LCI'],  forecast_with_exo['Pred_UCI'], alpha=0.3, color='g'
                     ,label='Confidence Interval')
    
    plt.title(f'AUTO ARIMA (*with* exogenous variables) - Actual vs Forecast for Store {store_num} and Product/Item {product_num}') 
    plt.legend(loc='upper left')
    plt.show()
    
    df_test_forecast = forecast_with_exo.copy().rename(columns={"Prediction":"Prediction_arima_exo", "Pred_LCI":"Pred_arima_exco_LCI",
                                                                "Pred_UCI":"Pred_arima_exco_UCI"})
    df_test_forecast = df_test_forecast.merge(test_with_exo[['store','item','sales']], how='inner', left_index= True,
                                              right_index = True)
    
    y_true = test_with_exo['sales']
    y_pred = forecast_with_exo['Prediction']
    y_true = np.array(y_true[:])
    y_pred = np.array(y_pred[:])
    
    MAPE_var = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)
    MSE_var = round(np.mean((np.square(y_pred - y_true))),2)
    RMSE_var = round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)
    
    print(f"MAPE for Store {store_num} and Product/Item {product_num} is: {MAPE_var}")
    print(f"MSE for Store {store_num} and Product/Item {product_num} is: {MSE_var}")
    print(f"RMSE for Store {store_num} and Product/Item {product_num} is: {RMSE_var} \n")
    
    df_accuracy = pd.DataFrame({'ID': ['Auto_SARIMAX_with_exo'], 'Store': [f'{store_num}'], 'Product':[f'{product_num}'],
                            'MAPE':[f'{MAPE_var}'],'MSE':[f'{MSE_var}'],'RMSE':[f'{RMSE_var}']}) 
    # return df_test_forecast, df_accuracy
    return df_test_forecast, df_accuracy, new_dict

df_forecast_1, df_accuracy_1, autoarima_with_exo_param_dict_1= auto_arima_with_exo(store_num = key_store_num_1, product_num = key_product_num_1, split_date = test_train_split_date)
df_forecast_2, df_accuracy_2, autoarima_with_exo_param_dict_2= auto_arima_with_exo(store_num = key_store_num_2, product_num = key_product_num_2, split_date = test_train_split_date)

df_autoarima_withexo_forecast = pd.concat([df_forecast_1,df_forecast_2])
df_autoarima_withexo_accuracy = pd.concat([df_accuracy_1,df_accuracy_2])

######################################################################################################################################## 
#                                                                                                                                      #    
#########################                                     XG BOOST                                      ############################ 
#                                                                                                                                      #
########################################################################################################################################

# store_num = key_store_num_1
# product_num = key_product_num_1
# test_train = 0.75
# time_delta = 150
# split_date = '2016-09-01'

# split_date_update = '2015-06-01'

def xg_boost (store_num, product_num, split_date, time_delta): 

    split_date_update = datetime.strptime(split_date, '%Y-%m-%d').date() + timedelta(days = - time_delta)
    print(split_date_update)
    
    
    df_xg_boost = df_arima_exo_update.copy()    
    df_xg_boost = df_xg_boost[(df_xg_boost['store'] == store_num) & (df_xg_boost['item'] == product_num)] 
    
    train_xg_boost = df_xg_boost[(df_xg_boost['date'] < f'{split_date}')]
    test_xg_boost = df_xg_boost[(df_xg_boost['date'] >= f'{split_date}')]
    
    y_train = train_xg_boost['sales']
    X_train = train_xg_boost.drop(columns = ['date','sales','store','item'])
    
    y_test = test_xg_boost['sales']
    X_test = test_xg_boost.drop(columns = ['date','sales','store','item'])
    
    model = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=123,
                 eval_metric = 'rmse')
    model.fit(X_train,y_train,eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50,verbose=False) 
    
    # _ = plot_importance(reg, height=0.9)
    
    plt.style.use('seaborn')
    plot_importance(model, height=0.9)
    
    fig, ax = plt.subplots(figsize=(20, 20))
    plot_tree(model, num_trees=20, ax=ax)
    plt.show()
    # plot_tree(model, num_trees=6, ax=ax)
    # plt.show()
    
    test_xg_boost['sales_Prediction'] = model.predict(X_test)
    test_xg_boost_all = pd.concat([train_xg_boost, test_xg_boost], sort=False)    
    
    # Plot Test, Train and Predition
    figure(figsize=(10,6))
    plt.style.use('seaborn-dark-palette')
    plt.plot(train_xg_boost['sales'][-100:], label='Train')
    plt.plot(test_xg_boost['sales'], label='Test')
    plt.plot(test_xg_boost['sales_Prediction'], label='Prediction')  
    plt.title(f'XGBoost - Actual vs Forecast for Store {store_num} and Product/Item {product_num}') 
    plt.legend(loc='upper left')
    plt.show()
    
    # test_xg_boost_all[(test_xg_boost_all['date'] > f'{split_date_update}')][['sales','sales_Prediction']].plot(figsize=(10, 6))
    
    df_test_forecast = test_xg_boost.copy().rename(columns={"sales_Prediction":"Prediction_xgboost"})
    df_test_forecast = df_test_forecast[['Prediction_xgboost']]
    df_test_forecast = df_test_forecast.merge(test_xg_boost[['store','item','sales']], how='inner', left_index= True,
                                              right_index = True)
    
    y_true = test_xg_boost['sales']
    y_pred = test_xg_boost['sales_Prediction']
    y_true = np.array(y_true[:])
    y_pred = np.array(y_pred[:])
    
    MAPE_var = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)
    MSE_var = round(np.mean((np.square(y_pred - y_true))),2)
    RMSE_var = round(np.sqrt(np.mean((np.square(y_pred - y_true)))),2)
    
    print(f"MAPE(XG_BOOST) for Store {store_num} and Product/Item {product_num} is: {MAPE_var}")
    print(f"MSE(XG_BOOST) for Store {store_num} and Product/Item {product_num} is: {MSE_var}")
    print(f"RMSE(XG_BOOST) for Store {store_num} and Product/Item {product_num} is: {RMSE_var} \n")
    
    df_accuracy = pd.DataFrame({'ID': ['XGBOOST'], 'Store': [f'{store_num}'], 'Product':[f'{product_num}'],
                            'MAPE':[f'{MAPE_var}'],'MSE':[f'{MSE_var}'],'RMSE':[f'{RMSE_var}']}) 
    
    xgboost_param_dict_1 = model.get_params
    
    # bst = xgb.Booster({'nthread': 4})  # init model
    # model_2 = bst.load_model('model.bin')  # load data
    
    return df_test_forecast, df_accuracy, xgboost_param_dict_1

df_forecast_1, df_accuracy_1, xgboost_param_dict_1 = xg_boost(store_num = key_store_num_1, product_num = key_product_num_1, split_date = test_train_split_date, time_delta = time_delta_var)
df_forecast_2, df_accuracy_2, xgboost_param_dict_2 = xg_boost(store_num = key_store_num_2, product_num = key_product_num_2, split_date = test_train_split_date, time_delta = time_delta_var)

df_xgboost_forecast = pd.concat([df_forecast_1,df_forecast_2])
df_xgboost_accuracy = pd.concat([df_accuracy_1,df_accuracy_2])

######################################################################################################################################## 
#                                                                                                                                      #    
#########################             COMBINE ALL THE FORECAST AND ACCURACY DATASETS TOGETHER               ############################ 
#                                                                                                                                      #
########################################################################################################################################

# Forecast
df_holt_winter_forecast = df_holt_winter_forecast.reset_index()
df_autoarima_withexo_forecast = df_autoarima_withexo_forecast.drop(columns = {'sales'}).reset_index()
df_autoarima_no_exo_forecast = df_autoarima_no_exo_forecast.drop(columns = {'sales'}).reset_index()
df_xgboost_forecast = df_xgboost_forecast.drop(columns = {'sales'}).reset_index()

data_frames = [df_holt_winter_forecast, df_autoarima_no_exo_forecast, df_autoarima_withexo_forecast, df_xgboost_forecast]
df_forecast_all = reduce(lambda  left,right: pd.merge(left,right,on=['date', 'store', 'item'], how='outer'), data_frames)

if output_forecast == "YES":
    df_forecast_all.to_excel(f'{output_path}Forecast_all.xlsx')
else:
    print("Forecast dataframe was *NOT* output to excel")

# Accuracy
df_accuracy_all = df_autoarima_withexo_accuracy.append([df_autoarima_no_exo_accuracy, df_holt_winter_accuracy,
                                                        df_xgboost_accuracy]).reset_index().drop(columns = {'index'})
df_accuracy_all.head(20)

if output_accuracy == "YES":
    df_accuracy_all.to_excel(f'{output_path}Accuracy_all.xlsx')
else:
    print("Accuracy dataframe was *NOT* output to excel")


########################################################### END OF CODE ################################################################
