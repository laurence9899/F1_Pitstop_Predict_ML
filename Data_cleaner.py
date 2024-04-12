#############################################################
# Filename: Data_cleaner.py
# Author: Laurence Hearne
# Last Modified: 13/02/2024
# Description: 
# Used to clean data.
#############################################################



# Imports
import pandas as pd                                                                 
from datetime import datetime
import os
from Data_clean_func import*

# Used to determine if folder for year being cleaned has already been created.
base_created=False

# Races per year
total_races2018 = 21
total_races2019 = 21
total_races2020 = 17
total_races2021 = 22
total_races2022 = 22
total_races2023 = 22

# Select year and race range
year=2023
races = list(range(1, total_races2023+1))
PitInTime=0
path=f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Modified_RaceData/{year}'

# Features which are to have interpoloation performed and features to be deleted
inter_features=['SpeedI1','SpeedI2','SpeedFL','SpeedST',
                'Sector1Time','Sector2Time','Sector3Time', 'LapTime']
delete_features=['DeletedReason','LapStartDate','LapStartTime','Sector1SessionTime',
                  'Sector2SessionTime','Sector3SessionTime','PitInTime','Rainfall']


# Create directory and move there
os.mkdir(path) 
os.chdir(path)

for race in races:

    file_path_RaceData = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/RaceData_ReadOnly/{year}/Race{race}_{year}.xlsx'
    file_path_Tyres = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Track_and Tyre_Data/{year}_Tyres.csv'

    isWet=False
    # Import race data from excel sheet
    df = pd.read_excel(file_path_RaceData, sheet_name='Sheet1')
    tyres = pd.read_csv(file_path_Tyres) 

    
    # If Intermediate or wet tyre used in race, label it as wet
    if (df== 'INTERMEDIATE').any().any() or (df== 'WET').any().any():
        
        isWet=True
    else:
        
        isWet=False

    # Loop through each row and apply the process_row function from Data_clean_func.py
    for index, row in df.iterrows():
        df.loc[index] =process_row(row)
    
    #Pass to function to complete changes to entire dataframe
    df=process_df(df,inter_features,delete_features,tyres,race,year)

    #-------------------------------------------------------------------------------
    #File saving and naming
    #-------------------------------------------------------------------------------
    #Below statment assigns wet and dry labels to filename.


    if isWet:
        new_filename=f'WET_CLEAN_RACE{race}_{year}.csv'
    else:    
        new_filename=f'DRY_CLEAN_RACE{race}_{year}.csv'
        
    print(new_filename)
    df.to_csv(new_filename, index=False)
    print(f'Round{race} cleaned sucessfully')







 
