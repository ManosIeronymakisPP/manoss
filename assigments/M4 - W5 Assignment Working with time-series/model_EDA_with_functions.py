# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 23:55:52 2023

@author: ManosIeronymakisProb
"""

#Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)

def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    """Merges three pandas DataFrames."""
    df1.Date = pd.to_datetime(df1.Date)
    df2.Date = pd.to_datetime(df2.Date)
    df1 = df1.merge(df3, on='Store')
    df = df1.merge(df2, on=['Store', 'Date', 'IsHoliday'])
    df = df.fillna(0)
    df = df.sort_values(by=['Date'])
    return df

# set the current working directory
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M4 - W5 Assignment Working with time-series")

# filepaths for the three datasets
filepath =  "Features data set.csv"
filepath1 =  "sales data-set.csv"
filepath2 =  "stores data-set.csv"

# load the three datasets into pandas dataframes
df_stores = load_data(filepath2)
df_features = load_data(filepath)
df_sales = load_data(filepath1)

# merge the three dataframes into a single dataframe
df = merge_dataframes(df_features, df_sales, df_stores)


def check_for_nan_values(df: pd.DataFrame) -> bool:
    """Checks if a pandas DataFrame has any NaN values.
    
    Args:
    df (pd.DataFrame): The pandas DataFrame to check.
    
    Returns:
    bool: True if the DataFrame has any NaN values, False otherwise.
    """
    if df.isna().any().any():
        print("There are NaN values in the DataFrame.")
        return True
    else:
        print("There are no NaN values in the DataFrame.")
        return False
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the given pandas DataFrame.
    
    Args:
    df (pd.DataFrame): The pandas DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed pandas DataFrame.
    """
    df = df.fillna(0)
    df['Year'] = df.Date.apply(lambda x: int(str(x)[:4]))
    df['Month'] = df.Date.apply(lambda x: int(str(x)[5:7]))
    df['Year-Month'] = df.Date.apply(lambda x: str(x)[:7])
    df['Day'] = df.Date.apply(lambda x: int(str(x)[8:10]))
    return df

def plot_fuel_price_over_time(df: pd.DataFrame, plot_no: int) -> None:
    """Plots the change in fuel price over the timeline of 3 years.
    
    Args:
    df (pd.DataFrame): The pandas DataFrame containing the data.
    plot_no (int): The number to use for the filename of the saved plot.
    """
    plt.subplots(figsize = (20,10))
    plt.xticks(rotation = 60)
    sns.lineplot(data = df, x = 'Year-Month',y = 'Fuel_Price')
    plt.title('LinePlot showing the change in fuel price over the span of 3 years', fontsize=20)
    plt.savefig(str(plot_no)+'_plot.png')

def plot_fuel_price(df: pd.DataFrame, r: int, plot_no: int) -> None:
    """
    Generate a barplot showing the change in fuel price with respect to the type of store with grouped holidays.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    r : int
        The rounding value for the temperature.
    plot_no : int
        The number to use as a suffix for the filename of the plot.
        
    Returns:
    --------
    None
    """
    # round off the temperature in the range of r
    df['Temperature_r'] = df.sort_values(by=['Temperature']).Temperature.apply(lambda x: x - x % r)

    # create the barplot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_ylim(3.1, 3.45)
    plots = sns.barplot(data=df, x='IsHoliday', y='Fuel_Price', hue='Type')
    ax.set_title('BarPlot showing the change in fuel price with respect the type of the store with holidays grouped')

    # add annotations to the bars
    for bar in plots.patches:
        ax.annotate(format(bar.get_height(), '.2f'),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() - (bar.get_height() - 3.1) / 2),
                    ha='center', va='center', size=15,
                    xytext=(0, 0),
                    bbox=dict(boxstyle="round4,pad=0.6", fc="w", ec="black", lw=2),
                    textcoords='offset points')

    # save the plot
    plt.savefig(f'{plot_no}_plot.png')

def fuel_price_lineplot(df: pd.DataFrame, plot_no: int) -> None:
    """
    Plot a lineplot showing the change in fuel price depending on the type of the store.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'Fuel_Price', 'IsHoliday', and 'Type' columns.
    plot_no (int): The index of the plot.

    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(20, 10))

    sns.lineplot(data=df, x='Type', y='Fuel_Price', hue='IsHoliday',
                 style='IsHoliday', markers=True, err_style='bars', ci=68, ax=ax)
    
    ax.set_title('LinePlot showing the change in fuel price with respect the type of the store')
    plt.savefig(f'{plot_no}_plot.png')

# There is a significant increase in the fuel price for the type B store and comparatively the fuels prices were very less during weekends.

def fuel_price_temp_lineplot(df: pd.DataFrame, plot_no: int) -> None:
    """
    Plot a lineplot showing the change in fuel price with respect to the change in temperature.
    
    Args:
    - df: a pandas DataFrame containing the data to be plotted
    - plot_no: an integer used to name the output file
    
    Returns: None
    """
    plt.subplots(figsize=(20,10))
    sns.lineplot(data=df, x='Temperature_r', y='Fuel_Price', hue='IsHoliday', style='IsHoliday', markers=True, errorbar=('ci', 68))
    plt.xlabel('Temperature range')
    plt.title('Lineplot showing the change in fuel price with respect to the change in temperature')
    plt.savefig(str(plot_no)+'_plot.png')
    plot_no += 1


def plot_cpi_temperature(df: pd.DataFrame, plot_no: int) -> None:
    """
    Generate a lineplot showing the change in CPI with respect to the change in temperature.

    Parameters:
    df (pd.DataFrame): A Pandas DataFrame containing the data.
    plot_no (int): The number of the plot.

    Returns:
    None.
    """
    plt.subplots(figsize = (20,10))
    sns.lineplot(data = df, x = 'Temperature_r', y = 'CPI', hue = 'Type',style = 'Type', markers = True, errorbar=('ci', 68))
    plt.title('Lineplot showing the change in CPI with respect to the change in temperature')
    plt.savefig(str(plot_no)+'_plot.png')
    plot_no += 1

#It is seen that the fuel price increases with increase in temperature steadily during workdays and unevenly during holidays


def plot_weekly_sales_by_month(df: pd.DataFrame) -> None:
    """
    Plot a line chart showing the change in weekly sales in each month over the span of 3 years.
    
    Parameters:
    df (pd.DataFrame): The pandas DataFrame containing the data to be plotted.
    
    Returns:
    None: This function saves the plot to a file and does not return anything.
    """
    global plot_no # global variable to keep track of the plot number
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(data=df, x='Year-Month', y='Weekly_Sales', ax=ax, ci=1)
    plt.xticks(rotation=60)
    plt.title('Lineplot showing the change in Weekly_Sales in each month over the span of 3 years')
    plt.savefig(str(plot_no) + '_plot.png') # save the plot to a file
    plot_no += 1 # increment the plot number for the next plot

#About seasonality there was a peak during the end of the years 2010 and 2011 but not during 2012. 
#This might be due to comparatively very less observations during the last 2 months in 2012.
#The most profitable weeks and months coincide with the holidays of Christmas and Thanksgiving which prooves that there are sessonality.

# call the function to check for NaN values
check_for_nan_values(df)

# preprocess the data
df = preprocess_data(df)

# plot the change in fuel price over the timeline of 3 years
plot_fuel_price_over_time(df, plot_no=1)

# plot the change in fuel price with respect to the type of store with grouped holidays
plot_fuel_price(df, r=0.02, plot_no=2)

# plot the change in fuel price depending on the type of the store
fuel_price_lineplot(df, plot_no=3)

# plot the weekly sales over time
fuel_price_temp_lineplot(df, plot_no=4)

# plot the distribution of weekly sales
plot_cpi_temperature(df, plot_no=5)

# plot the correlation heatmap
plot_weekly_sales_by_month(df, plot_no=6)





