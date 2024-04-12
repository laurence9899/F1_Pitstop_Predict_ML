#############################################################
# Filename: M10_8_2_24.py
# Author: Laurence Hearne
# Last Modified: 16/03/2024
# Description: 
# Create and train tensorflow model. 
# Hyperparamters set to optimal ones determined in hyperparmater tuning.
# PitStopBehind feature added.
# Test trainied model on test set
#############################################################



#Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import*
from tensorflow.keras.utils import to_categorical
from build_df_raceData import*
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model



# data import
df = build_df_years_processed(2018, 2023)


#Replace any infinite values with nan, remove nans
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()



#Sort by Driver and Race_ID
df = df.sort_values(['Race_ID', 'Driver','Race_Progress'])


#Save dataset to csv
df.to_csv("dataset_2018_2023.csv")


#One hot encode Relative Compound, track type and track status
encoder = OneHotEncoder(sparse=False)
compound_encoded = encoder.fit_transform(df[['Relative_Compound']])
track_type_encoded = encoder.fit_transform(df[['Track_Type']])
track_status_encoded = encoder.fit_transform(df[['TrackStatus']])


# Convert boolean columns to integers
df['CarClose'] = df['CarClose'].astype(int)
df['InPits'] = df['InPits'].astype(int)
df['TwoCompoundsUsed'] = df['TwoCompoundsUsed'].astype(int)
df['PitstopBehind'] = df['PitstopBehind'].astype(int)



# Create list of model deatures from dataset. Note RaceID used for divding up data in test, train validation.
# RaceID will not be passed to model.
features = [
    'CarClose','TwoCompoundsUsed', 'Race_Progress', 'TyreLife_Progress',
    'N_Driver_Champ_Pos', 'N_Position','Race_ID','PitstopBehind'
]


#Select all features from dataseta dn create new datframe
X = pd.concat([
    pd.DataFrame(compound_encoded).reset_index(drop=True),
    pd.DataFrame(track_type_encoded).reset_index(drop=True),
    pd.DataFrame(track_status_encoded).reset_index(drop=True),
    df[features].reset_index(drop=True)
], axis=1)

# Create dataframe of predictor variable, RaceID, LapID and Driver used to
# for easy analysis of results. On Inpits will be passed to model.
y = df[['InPits','Race_ID','Lap_ID','Driver']].copy()



# Split the dataset into training, validation and testing sets
X_train=X[X['Race_ID']<=202300]#Select all races between 2018 and last race of 2022
y_train=y[y['Race_ID']<=202300]

X_val=X[X['Race_ID']>202300]#Select 2023 season
X_val=X_val[X_val['Race_ID']<202312]#Select first 10 races of 2023 as vaildation data

y_val=y[y['Race_ID']>202300]#Select 2023 season
y_val=y_val[y_val['Race_ID']<202312]#Select first 10 races of 2023 as vaildation data

X_test=X[X['Race_ID']>=202312]#Select Last 10 races of 2023 as test data
y_test=y[y['Race_ID']>=202312]


# Save test and training sets to csv
X_test.to_csv('X_Test.csv')
y_test.to_csv('y_test.csv')
X_train.to_csv('X_train.csv')


#Race ID used for dividing up test and training data, RaceID, LapID and Driver not required for training so can be dropped
X_train=X_train.drop(columns='Race_ID')
X_test=X_test.drop(columns='Race_ID')
X_val=X_val.drop(columns='Race_ID')
y_train=y_train.drop(columns=['Race_ID','Lap_ID','Driver'])
y_test=y_test.drop(columns=['Race_ID','Lap_ID','Driver'])
y_val=y_val.drop(columns=['Race_ID','Lap_ID','Driver'])


#Reset indeices of dataframes and drop index as it will have been duplicated.
X_train=X_train.reset_index()
y_test=y_test.reset_index()
X_test=X_test.reset_index()
X_val=X_val.reset_index()
y_val=y_val.reset_index()
X_train = X_train.drop(columns=['index'])
X_test = X_test.drop(columns=['index'])
X_val=X_val.drop(columns=['index'])
y_val=y_val.drop(columns=['index'])
y_test = y_test.drop(columns=['index'])

# Build the neural network model. Hyperparmaters found during Hyp par opt run
model = Sequential()
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.0003)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0003)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])



# Define class weights
class_weights = {0: 1, 1:21}


# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')



#Train the model
model.fit(X_train, y_train,
          class_weight=class_weights,
          batch_size=128,
          epochs=100,  
          validation_data=(X_val, y_val), 
          verbose=1,
          callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Print model structure
model.summary()

# Produce predictions on the test set.
predictions = model.predict(X_test)
print(type(predictions))
pred = pd.DataFrame(predictions)
pred.to_clipboard()

print("Run Complete")

# Save the model
model.save('M10')