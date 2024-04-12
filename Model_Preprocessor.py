#############################################################
# Filename: Model_preprocessor.py
# Author: Laurence Hearne
# Last Modified: 15/02/2024
# Description: 
# Used to process data peioer to model training but after data cleaning.
# Actions performed here are infromed by EDA.
# Uses several functions from script preprocessor_func
#############################################################




# Imports
import pandas as pd
from datetime import datetime
import os
from build_df_raceData import*
from preprocessor_func import*
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Races in each season
total_races2018 = 21
total_races2019 = 21
total_races2020 = 17
total_races2021 = 22
total_races2022 = 22
total_races2023 = 22


#Preprocessing settings 
years=range(2023,2023+1)
columns_to_normalize = ['LapNumber', 'TyreLife','Driver_Champ_Pos','Position','NoLapsToPit','GapBehind','GapAhead','GapToLeader']
normalized_coloumns = ['Race_Progress', 'TyreLife_Progress','N_Driver_Champ_Pos','N_Position','N_Laps_to_Pit','N_GapBehind','N_GapAhead','N_GapToLeader']
minlap=2
maxlap=2
unique_value = 0

# Select race range
races = list(range(1, 23))

# Loop through years
for year in years:
    driver_champ_df = pd.read_csv(f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Drivers_Champ{year}.csv')
    constr_champ_df = pd.read_csv(f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Const_Champ{year}.csv')
    path=f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Dataset/{year}'
    try:
        os.mkdir(path) 
    except:
        print('Folder Already Exists')
    os.chdir(path)
    # Loop through races
    for race in races:
          try:
            #Read in data
            file_path_RaceData = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Verified_Racedata/{year}/V_Race{race}_{year}.csv'
            df = pd.read_csv(file_path_RaceData)
            print("Race found")
        
            median_laptime = (df['LapTime'].median())/1000 

            # Replace the 'L' laps with their corresponding multiples of the median lap time
            df.replace(to_replace='1 L', value=median_laptime, regex=True, inplace=True)
            df.replace(to_replace='2 L', value=(2 * median_laptime), regex=True, inplace=True)
            df.replace(to_replace='3 L', value=(3 * median_laptime), regex=True, inplace=True)
            df.replace(to_replace='4 L', value=(4 * median_laptime), regex=True, inplace=True)

            df.replace(to_replace='1L', value=median_laptime, regex=True, inplace=True)
            df.replace(to_replace='2L', value=(2 * median_laptime), regex=True, inplace=True)
            df.replace(to_replace='3L', value=(3 * median_laptime), regex=True, inplace=True)
            df.replace(to_replace='4L', value=(4 * median_laptime), regex=True, inplace=True)
            df.replace(to_replace='10L', value=(10 * median_laptime), regex=True, inplace=True)

            #Remove drivers laps with TimeInPits of greater than 50 seconds
            # median_time_in_pits = df[df['TimeInPits'] > 0]['TimeInPits'].median()
            # print(median_time_in_pits*1.75)
            drivers_to_remove = df[df['TimeInPits'] >(50)]['Driver'].unique()
            print(race)
            print(drivers_to_remove)
            df = df[~df['Driver'].isin(drivers_to_remove)]
            df['GapAhead'].to_clipboard()
            #Remove laps of driver with a laptime of greater than 150000 milliseconds
            drivers_to_remove = df[df['LapTime'] > 150000]['Driver'].unique()
            print(drivers_to_remove)
            df = df[~df['Driver'].isin(drivers_to_remove)]

            #Remove laps of Driver with more than 4 pitstops
            drivers_to_remove = df[df['NumberOfPitStops'] > 4]['Driver'].unique()
            print(drivers_to_remove)
            df = df[~df['Driver'].isin(drivers_to_remove)]

            #Insert Championship position position for driver and teams
            df = df.merge(driver_champ_df, left_on='Driver', right_on='Driver', how='left')
            df = df.merge(constr_champ_df, left_on='Team', right_on='Team', how='left')

            #Check if two tyre compounds have been used by each driver
            df=calculate_two_compounds_used(df)
            
            #Calculate the number of laps until the next pitstop
            df=calculate_nolapstopit(df)
            print('checkpoint1')
            #Sort by Lapnumber and psoition
            df = df.sort_values(['LapNumber', 'Position'])
            df['GapAhead'] = df['GapAhead'].astype(float)
            #Car close Ahead
            df['CloseAhead'] = (df['GapAhead'] < 2) & (df['GapAhead']!=0)
            
            #Calculate Gap behind
            df['GapBehind'] = df['GapAhead'].shift(-1)
            print('checkpoint2')
            #Car closed behind
            df['CloseBehind'] = (df['GapBehind'] < 2) & (df['GapBehind']!=0)

            #Car Close
            df['CarClose']=df['CloseBehind'] | df['CloseAhead']
            
            #Divide track status up into 5 categories
            #0: Red Flag
            #1: Green Flag
            #2: Yellow Flags
            #3: VSC deployed
            #4: Safety Car Deployed
            df['TrackStatus'] = df['TrackStatus'].apply(transform_track_status)
            
            #Normalise Values
            df=normalize_features(df,columns_to_normalize,normalized_coloumns)
            #Remove first and last lap
            df=remove_laps(df,minlap,(df['LapNumber'].max()-maxlap))
            
            

            
            df = ad_pitstop_behind_feature(df)
            #Add Race ID
            if(race>9):
                df["Race_ID"]=f"{year}{race}"
                df["Race_ID"]=df["Race_ID"].astype(int)
            else:
                df["Race_ID"]=f"{year}0{race}"
                df["Race_ID"]=df["Race_ID"].astype(int)

            #Add Lap_ID
            # Initialize a unique value, for example, a counter
            

            for index, row in df.iterrows():
                df.at[index, 'Lap_ID'] = unique_value
                unique_value += 1  # Increment the unique value

            #Save dataframe to a File
            df.to_csv(f'P_Race{race}_{year}.csv', index=False)

          except:
            print("Race not found")
