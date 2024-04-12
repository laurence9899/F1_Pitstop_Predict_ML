#############################################################
# Filename: aquire_timingdata.py
# Author: Laurence Hearne
# Last Modified: 29/01/2024
# Description: 
# Acesses fastf1 api to download data on the gaps between drivers
# Merges with dataset
#############################################################


# Imports
import fastf1
import fastf1.api
import openpyxl
import numpy as np
import time
import pandas as pd

# Set up cache for fastf1
fastf1.Cache.enable_cache('/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/FastF1_Cache/')

# Get gap to leader for driver
def _get_gap_str_for_drv(drv, idx, laps_data, stream_data):
    first_time = laps_data[laps_data['Driver'] == drv].iloc[idx]['Time']
    ref_idx = (stream_data[stream_data['Driver'] == drv]['Time']
               - first_time).abs().idxmin()
    gap_str = stream_data.loc[ref_idx]['GapToLeader']
    return gap_str

# Get gap ahead for a driver
def _get_gap_for_drv(drv, idx, laps_data, stream_data):
    first_time = laps_data[laps_data['Driver'] == drv].iloc[idx]['Time']
    ref_idx = (stream_data[stream_data['Driver'] == drv]['Time']
               - first_time).abs().idxmin()
    gap_str = stream_data.loc[ref_idx]['IntervalToPositionAhead']
    return gap_str


# Function to check if a cell contains "LAP X"
def check_lap(val):
    if isinstance(val, str) and val.startswith("LAP"):
        return True
    return False

# Removes + sign to ensure value can be convterted to float
def clean_value(val):
    if check_lap(val):
        return 0.0  # Replace "LAP X" with zero
    return float(val.replace('+', '')) if isinstance(val, str) else val



# Determine the tiem spent in pits and add it as new column in df
def calculate_pit_time(df):

    # Create a new column to hold the pit time difference
    df['TimeInPits'] = 0

    # Iterate through the DataFrame
    for i in range(1, len(df)):
        # Check if the current row is the PitInTime for the previous PitOutTime
        pit_time = df.loc[i, 'PitOutTime'] - df.loc[i-1, 'PitInTime']
        df.loc[i, 'TimeInPits'] = pit_time.total_seconds()
       
    df['TimeInPits']=df['TimeInPits'].fillna(0)
    df['TimeInPits'] = df['TimeInPits'].apply(lambda x: max(min(x, 150), 0))


    return df



# Races per season
total_races2018 = 21
total_races2019 = 21
total_races2020 = 17
total_races2021 = 22
total_races2022 = 22
total_races2023 = 22

# Select year and race range
year=2023
races = list(range(1, total_races2023 +1))

# Loop through each row in the DataFrame
for race in races:
    

    # Load lap and race data
    raceData = fastf1.get_session(year, race, 'r')
    [laps_data,stream_data] = fastf1.api.timing_data(raceData.api_path)


    # Assuming laps_data and stream_data are your DataFrames with the race data
    unique_drivers = laps_data['Driver'].unique()
    max_laps = laps_data['NumberOfLaps'].max()

    # Create an empty DataFrame to store the results
    gaps_df = pd.DataFrame()

    for driver in unique_drivers:
        for lap in range(1, max_laps + 1):
            if lap in laps_data[laps_data['Driver'] == driver]['NumberOfLaps'].values:
                gap_leader = _get_gap_str_for_drv(driver, lap - 1, laps_data, stream_data)
                gap_ahead = _get_gap_for_drv(driver, lap - 1, laps_data, stream_data)
                gaps_df = gaps_df._append({'DriverNumber': driver, 'LapNumber': lap, 'GapToLeader': gap_leader,'GapAhead': gap_ahead}, ignore_index=True)

    # Now gaps_df contains the gap to the leader for each driver at each lap




    file_path_RaceData = f'/Users/laurencehearne/Library/CloudStorage/OneDrive-UniversityofLimerick/FYP/Race_Data/Modified_RaceData/{year}/DRY_CLEAN_RACE{race}_{year}.csv'
    try:
        df = pd.read_csv(file_path_RaceData)
        print("Race Found")
        # Set 'LapNumber' and 'DriverNumber' as indices
        # Set the index of both DataFrames to 'LapNumber' and 'Driver'
        gaps_df["DriverNumber"] = gaps_df["DriverNumber"].astype(int)
        gaps_df["LapNumber"] = gaps_df["LapNumber"].astype(int)
        No_PitStops=laps_data["NumberOfPitStops"].astype(int)
        laps_data=calculate_pit_time(laps_data)
        gaps_df = pd.concat([gaps_df,No_PitStops,laps_data['TimeInPits']], axis=1)


        # Replace 'LAP X' with zero
        gaps_df.replace(to_replace=r'^LAP \d+$', value=0.0, regex=True, inplace=True)

        # Convert all other values to floats, stripping the '+' sign
        gaps_df = gaps_df.applymap(lambda x: float(x.strip('+')) if isinstance(x, str) and x.startswith('+') else x)


        df["DriverNumber"] = df["DriverNumber"].astype(int)
        df["LapNumber"] = df["LapNumber"].astype(int)
        df.set_index(['LapNumber', 'DriverNumber'], inplace=True)
        gaps_df.set_index(['LapNumber', 'DriverNumber'], inplace=True)

        # Sort both DataFrames by index to ensure alignment
        df.sort_index(inplace=True)
        gaps_df.sort_index(inplace=True)

        # Concatenate the DataFrames side by side
        concatenated_df = pd.concat([df, gaps_df], axis=1)

        # Reset the index if you want 'LapNumber' and 'Driver' back as columns
        concatenated_df.reset_index(inplace=True)
            
        concatenated_df = concatenated_df.sort_values(['LapNumber', 'Position'])
        concatenated_df.to_csv(f"V_Race{race}_{year}.csv")
    except:
        print("Error in Try Block")

    race=race+1



