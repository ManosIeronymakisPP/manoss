
#Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M4 - W5 Assignment Working with time-series")
filepath =  "Features data set.csv"
filepath1 =  "sales data-set.csv"
filepath2 =  "stores data-set.csv"

# Loading the data into pandas dataframe for EDA
df_stores = pd.read_csv(filepath2)
df_features = pd.read_csv(filepath)
df_features.Date = pd.to_datetime(df_features.Date)
df_sales = pd.read_csv(filepath1)
df_sales.Date = pd.to_datetime(df_sales.Date)

#merging the dataframes
df_features = df_features.merge(df_stores, on = 'Store')
df = df_features.merge(df_sales, on = ['Store','Date','IsHoliday'])
df=df.fillna(0)
df = df.sort_values(by = ['Date'])


# check if any column has NaN values 
if df.isna().any().any():
    print("There are NaN values in the DataFrame.")
else:
    print("There are no NaN values in the DataFrame.")
    
#remove teh nan values with 0 
df=df.fillna(0)

df.describe()
df.info()

# splitting date into 3 columns denoting Year, Month and Day respectively
df['Year'] = df.Date.apply(lambda x: int(str(x)[:4]))
df['Month'] = df.Date.apply(lambda x: int(str(x)[5:7]))
df['Year-Month'] = df.Date.apply(lambda x: str(x)[:7])
df['Day'] = df.Date.apply(lambda x: int(str(x)[8:10]))

#lineplot showing the change in fuel price over the timeline of 3 years
plot_no = 1
_ = plt.subplots(figsize = (20,10))
_ = plt.xticks(rotation = 60)
_ = sns.lineplot(data = df, x = 'Year-Month',y = 'Fuel_Price')
_ = plt.title('LinePlot showing the change in fuel price over the span of 3 years', fontsize=20)
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1

#barplot showing the change in fuel price with respect the type of the store with grooped holidays
r = 5 #lets round off the temperature in the range of r
df['Temperature_r'] = df.sort_values(by=['Temperature']).Temperature.apply(lambda x : x - x %r)

_ = plt.subplots(figsize = (20,10))
_ = plt.ylim(3.1,3.45)
plots = sns.barplot(data = df, x = 'IsHoliday', y = 'Fuel_Price', hue = 'Type')
_ = plt.title('BarPlot showing the change in fuel price with respect the type of the store with holidays grouped')
for bar in plots.patches:
    _ = plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() - (bar.get_height()-3.1)/2), ha='center', va='center',
                   size=15, xytext=(0, 0),bbox=dict(boxstyle="round4,pad=0.6", fc="w", ec="black", lw=2),
                   textcoords='offset points')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1

#lineplot showing the change in fuel price depending the type of the store
_ = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Type', y = 'Fuel_Price', hue = 'IsHoliday',style = 'IsHoliday', markers = True, errorbar=('ci', 68))
_ = plt.title('LinePlot showing the change in fuel price with respect the type of the store')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1

# There is a significant increase in the fuel price for the type B store and comparatively the fuels prices were very less during weekends.

#lineplot showing the change in fuel price with respect to the change in temperature
_ = plt.subplots(figsize=(20,10))
_ = sns.lineplot(data=df, x='Temperature_r', y='Fuel_Price', hue='IsHoliday', style='IsHoliday', markers=True, errorbar=('ci', 68))
_ = plt.xlabel('Temperature range')
_ = plt.title('Lineplot showing the change in fuel price with respect to the change in temperature')
plt.savefig(str(plot_no)+'_plot.png')
plot_no += 1

#lineplot showing the change in CPI with respect to the chnage of temperature
_ = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Temperature_r', y = 'CPI', hue = 'Type',style = 'Type', markers = True, errorbar=('ci', 68))
_ = plt.title('Lineplot showing the change in CPI with respect to the change in temperature')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1

#It is seen that the fuel price increases with increase in temperature steadily during workdays and unevenly during holidays

#lineplot showing the change in fuel price in each month over the timeline of 3 years
_ = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Date', y = 'Fuel_Price')
_ = plt.title('Lineplot showing the change in fuel price in each month over the span of 3 years')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1


#lineplot showing the change in weekly sales in each month
fig,ax = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Year-Month', y = 'Weekly_Sales', ax = ax, ci = 1)
_ = plt.xticks(rotation = 60)
_ = plt.title('Lineplot showing the change in Weekly_Sales in each month over the span of 3 years')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1

#About seasonality there was a peak during the end of the years 2010 and 2011 but not during 2012. 
#This might be due to comparatively very less observations during the last 2 months in 2012.
#The most profitable weeks and months coincide with the holidays of Christmas and Thanksgiving which prooves that there are sessonality.



