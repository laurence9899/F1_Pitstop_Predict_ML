#############################################################
# Filename: EDA.py
# Author: Laurence Hearne
# Last Modified: 29/03/2024
# Description: 
# Used to do various plots for EDA pruposes
#############################################################

# Imports
import pandas as pd
from datetime import datetime
import os
from pandas_profiling import ProfileReport
import sweetviz as sv
import dtale
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from build_df_raceData import*

# Select years
year_start=2018
year_end=2023
delete=['FastF1Generated','Unnamed: 0']
df=build_df_years_processed(year_start,year_end)
df.drop(columns=delete,axis=1, inplace=True)


# Filter out rows where 'TimeInPits' is zero
filtered_df = df[df['LapTime'] != 0]

# Plotting the histogram for the 'TimeInPits' column of the filtered DataFrame
plt.hist(filtered_df['LapTime'], bins=100, edgecolor='black',log=True)

# Adding titles and labels
plt.title('Histogram of Time in Pits for Raw Data')
plt.xlabel('Time in Pits(Seconds)')
plt.ylabel('Frequency(Log Scale)')

plt.show()

# Process the data to find the maximum number of pitstops for each driver per race
grouped_data = df.groupby(['Driver', 'Race_ID'])['NumberOfPitStops'].max()

# Reset index to convert grouped data back to DataFrame
max_pitstops = grouped_data.reset_index()

# Now, find the maximum pitstops for each driver across all races
final_data = max_pitstops.groupby('Driver')['NumberOfPitStops'].max()
binedges=list(range(0,8))
# Plotting a histogram
plt.figure(figsize=(10,6))
plt.hist(max_pitstops['NumberOfPitStops'], bins=binedges, alpha=0.7, color='blue',edgecolor='black')
plt.title('Histogram Number of Pitstops per Driver Across All Races(Raw Data)')
plt.xlabel('Number of Pitstops')
plt.ylabel('Frequency')
plt.show()





#EDA using pandas-profiling
profile=ProfileReport(df, explorative=True)
#Saving results to a HTML file
profile.to_file("ALL_Races_pandas_2020.html")
#EDA using Autoviz
sweet_report = sv.analyze(df)
#Saving results to HTML file
sweet_report.show_html('All_Races_sweet_2018_2023.html')


#Corelation Plots

corr_matrix = df.select_dtypes(include='number').corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()



ax = sn.scatterplot(x="Track_Type", y="Sector3Time", data=df)
ax.set_title("Sector3Time vs. Track_Type")
ax.set_xlabel("Fly ash")
plt.show()
sn.lmplot(x="Track_Type", y="Sector3Time", data=df)


# Create a line plot
plt.scatter(df['Round'],df['WindSpeed'])
plt.xlabel('round')
plt.ylabel('WindSpeed(m/s)')
plt.title('Scatter plot of WindSpeed Across the Rounds of 2021')
plt.xticks(np.arange(0,22,2))
# Show the plot
plt.show()