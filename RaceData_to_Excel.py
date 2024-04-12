#############################################################
# Filename: RaceData_to_Excel.py
# Author: Laurence Hearne
# Last Modified: 20/12/2023
# Description: 
# Used to download Racedata from fastf1 library and save to excel file.
#############################################################


#Imports
import fastf1
import openpyxl
import numpy as np
import time
import pandas as pd

# Set up cache for fastf1
fastf1.Cache.enable_cache('/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/FastF1_Cache/')

# List of race Numbers for a season i.e. from 1 to total number of races in the season

total_races2018 = 21
total_races2019 = 21
total_races2020 = 17
total_races2021 = 22
total_races2022 = 22


# Specify year and raneg of races to download
year=2023
races = list(range(1, total_races2018+1))
print(races)



# for loop to cycle through each race of a season and save off race data to an Excel sheet
# First loop used to save races to cache
for race in races:
    # Create session object using fasf1
    raceData = fastf1.get_session(year, race, 'r')
    # Load laps
    raceData.load(laps=True,weather=True)

for race in races:
    # Create session object
    raceData = fastf1.get_session(year, race, 'r')
    # Load laps
    raceData.load(laps=True,weather=True)
    # Load weather data
    weather_data = raceData.laps.get_weather_data()
    lap_data=raceData.laps
    lap_data = lap_data.reset_index(drop=True)
    weather_data = weather_data.reset_index(drop=True)
    # Join weather and lap data
    joined = pd.concat([lap_data, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1)
    # Save off race data s .xlsx
    joined.to_excel(f"Race{race}_{year}.xlsx")

