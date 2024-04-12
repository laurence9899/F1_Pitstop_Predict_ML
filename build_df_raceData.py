#############################################################
# Filename: build_df_raceData.py
# Author: Laurence Hearne
# Last Modified: 20/01/2024
# Description: 
# Contaisn two functiosn for reacding in all individual racedat files and combining 
# them into a single dataframe
#############################################################

# Imports
import pandas as pd

total_races2018 = 21
total_races2019 = 21
total_races2020 = 17
total_races2021 = 22
total_races2022 = 22
total_races2024 = 22



# initialize list of lists 
races_per_year = [21,21,17,22,22,22]
# year            18,19,20,21,22,23
 

year=2023
print(races_per_year[year-2018])

# Build a dataframe of all processed races
def build_df_years_processed(year_start,year_end):
    df = pd.DataFrame()

    years=range(year_start,year_end+1)
    print(years)
    for year in years:
        races=range(1,races_per_year[year-2018]+1)
        print(races)
        for race in races:
            try:
                file_path = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Dataset/{year}/P_Race{race}_{year}.csv'
                print(file_path)
                df_tmp = pd.read_csv(file_path) 
                df=pd.concat([df,df_tmp])
            except:
                print('Race Not Found')

    return df

# Build a dataframe of all un processed race
def build_df_years_verified(year_start,year_end):
    df = pd.DataFrame()

    years=range(year_start,year_end+1)
    print(years)
    for year in years:
        races=range(1,races_per_year[year-2018]+1)
        print(races)
        for race in races:
            try:
                file_path = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Verified_RaceData/{year}/V_Race{race}_{year}.csv'
                print(file_path)
                df_tmp = pd.read_csv(file_path) 
                df=pd.concat([df,df_tmp])
            except:
                print('Race Not Found')

    return df


def build_df_years_neutral_insert(year_start,year_end):
    df = pd.DataFrame()

    years=range(year_start,year_end+1)
    print(years)
    for year in years:
        races=range(1,races_per_year[year-2018]+1)
        print(races)
        for race in races:
            try:
                file_path = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Verified_RaceData/{year}/V_Race{race}_{year}.csv'
                print(file_path)
                df_tmp = pd.read_csv(file_path) 
                
                df=pd.concat([df,df_tmp])
            except:
                print('Race Not Found')

    return df