#############################################################
# Filename: Data_cleaner_func.py
# Author: Laurence Hearne
# Last Modified: 13/02/2024
# Description: 
# Contains various functions used by the Data_cleaner.py script.
#############################################################


import pandas as pd
from datetime import datetime




#Define function that will perfrom necessary actions to clean each row.
#Inputs: row of dataframe
#Returns: modified row of data.
def process_row(row):
    
    global isWet
    #Convert LapTime, Sector1Time, Sector2Time and Sector3Time to milliseconds
    row['LapTime'] = row['LapTime'] * 86400000
    row['Sector1Time'] = row['Sector1Time'] * 86400000
    row['Sector2Time'] = row['Sector2Time'] * 86400000
    row['Sector3Time'] = row['Sector3Time'] * 86400000
    row['PitInTime'] = row['PitInTime'] * 86400000
    row['PitOutTime'] = row['PitOutTime'] * 86400000 
    #row['TimeInPits']=   
    #Identify laps driver entered pits and create new feature called 'InPits' to stor
    row['PitOutTime']=not(pd.isna(row['PitInTime']))
    


    return row

# Function applies actions relavent to the entire dataframe
def process_df(df,inter_features,delete_features, tyres,round,year ):
    # Create new abolute compound column
    df_temp = pd.DataFrame()
    df_temp['Relative_Compound']=df['Compound']
    df.rename(columns={'PitOutTime': 'InPits'}, inplace=True)
    df.drop(columns=delete_features,axis=1, inplace=True)
    # Use ffill to interpolate slected features
    df[inter_features]=df[inter_features].interpolate(method='ffill')
    df_compound=pd.DataFrame()
    
    # Depending on year select different compounds to replace
    df=df.replace('HARD', tyres['Hard'][round-1])
    df=df.replace('MEDIUM', tyres['Medium'][round-1])
    df=df.replace('SOFT', tyres['Soft'][round-1])
    # df=df.replace('HYPERSOFT', 'SSSS')
    # df=df.replace('ULTRASOFT', 'SSS')
    # df=df.replace('SUPERSOFT', 'SS')
    # df=df.replace('SOFT', 'S')
    # df=df.replace('MEDIUM', 'H')
    # df=df.replace('HARD', 'HHH')
    df_compound['Compound']=df['Compound']
    df=df.replace(tyres['Hard'][round-1],'HARD')
    df=df.replace(tyres['Medium'][round-1],'MEDIUM')
    df=df.replace( tyres['Soft'][round-1],'SOFT')
    df_compound['Relative_Compound']=df['Compound']
    
    df['Relative_Compound']=df_compound['Relative_Compound']
    df['Compound']=df_compound['Compound']
    raceTrack=tyres.loc[round-1] 
    df['Year']=year
    df['Round']=round
    # Add track type
    type=raceTrack['Type']
    df['Track_Type']=type
    
    return df