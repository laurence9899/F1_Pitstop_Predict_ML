#############################################################
# Filename: preprocessor_func.py
# Author: Laurence Hearne
# Last Modified: 15/02/2024
# Description: 
# Contains various functions used in Model_Preprocessor.py
#############################################################



# Imports
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


#Function to normalsie features
def normalize_features(df, columns_to_normalize,normalized_coloumns):
    # Create a copy of the DataFrame to avoid modifying the original
    normalized_df = df.copy()

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the specified columns
    normalized_df[normalized_coloumns] = scaler.fit_transform(df[columns_to_normalize])

    return normalized_df


# Remove specified early and late race laps
def remove_laps(race_data,minlap,maxlap):
    # Remove the first two laps for each driver
    filtered_race_data = race_data[race_data['LapNumber'] > minlap]
    filtered_race_data = filtered_race_data[race_data['LapNumber'] <maxlap]
    return filtered_race_data

def calculate_two_compounds_used(df, driver_col='Driver', lap_col='LapNumber', compound_col='Compound'):

    # Sort the DataFrame by Driver and LapNumber to ensure chronological order
    df.sort_values(by=[driver_col, lap_col], inplace=True)

    # Initialize the new column to False
    df['TwoCompoundsUsed'] = False

    # Dictionary to keep track of the compounds used by each driver
    compounds_used = {}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        driver = row[driver_col]
        compound = row[compound_col]
        
        # Initialize the set for the driver if not present
        if driver not in compounds_used:
            compounds_used[driver] = set()
        
        # Add the current compound to the set
        compounds_used[driver].add(compound)
        
        # Check if more than one compound has been used up to this lap
        df.at[index, 'TwoCompoundsUsed'] = len(compounds_used[driver]) > 1

    return df


# Used for regression model,
# Not used in M10
def calculate_nolapstopit(df):
    # Sort values by Driver and LapNumber to ensure correct sequential processing
    df_sorted = df.sort_values(by=['Driver', 'LapNumber'])

    # Initialize the new column with default values
    df_sorted['NoLapsToPit'] = 0

    # Iterate through each driver
    for driver in df_sorted['Driver'].unique():
        # Filter dataframe for the current driver
        driver_df = df_sorted[df_sorted['Driver'] == driver]

        # Find laps where the driver pits
        pit_laps = driver_df[driver_df['InPits']]['LapNumber'].tolist()

        # Initialize the next pit lap to the first pit lap or a large number if no pit stops
        next_pit_lap = pit_laps[0] if len(pit_laps) > 0 else max(df_sorted['LapNumber']) + 1

        # Iterate through each lap in order
        for index, row in driver_df.iterrows():
            current_lap = row['LapNumber']
            if current_lap in pit_laps:  # Update next pit lap when a pit stop is encountered
                next_pit_index = pit_laps.index(current_lap) + 1
                next_pit_lap = pit_laps[next_pit_index] if next_pit_index < len(pit_laps) else max(df_sorted['LapNumber']) + 1
            
            # Calculate laps to next pit stop, set to 0 if current lap is a pit lap
            df_sorted.at[index, 'NoLapsToPit'] = 0 if current_lap in pit_laps else next_pit_lap - current_lap

    return df_sorted


# Function to ensure that onlt 5 track catgorys present in dataset
def transform_track_status(row):
    # Check if the row contains 4, 5, or 6 and return the respective values
    if '4' in str(row):
        return 4
    elif '5' in str(row):
        return 0
    elif '6' in str(row):
        return 3
    else:
        return row  # Return the original value if none of the conditions are met

# Add new PitStopBehind feature.
# Feature details explained in report

def ad_pitstop_behind_feature(df):
    # Ensure the DataFrame is sorted by LapNumber and Position to accurately determine positions
    df = df.sort_values(by=['LapNumber', 'Position'])
    
    # Initialize the new column with False for all rows
    df['PitstopBehind'] = False
    
    # Iterate over the DataFrame to set 'PitstopBehind' for the next two laps
    for index, row in df.iterrows():
        # Look for the car directly behind in the same lap
        car_behind = df[(df['LapNumber'] == row['LapNumber']) & (df['Position'] == row['Position'] + 1)]
        if not car_behind.empty:
            # Check if the car behind pitted
            if car_behind.iloc[0]['InPits']:
                # Find the driver's record in the next lap and the lap after next
                for next_lap_offset in [1, 2]:
                    next_lap_record = df[(df['LapNumber'] == row['LapNumber'] + next_lap_offset) & (df['Driver'] == row['Driver'])]
                    if not next_lap_record.empty:
                        # Set 'PitstopBehind' to True for the next two laps
                        df.at[next_lap_record.index[0], 'PitstopBehind'] = True
    
    return df












