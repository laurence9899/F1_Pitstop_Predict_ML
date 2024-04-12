#############################################################
# Filename: Hyperparam_opt.py.py
# Author: Laurence Hearne
# Last Modified: 20/02/2024
# Description: 
# Used to run hyperparmater tuning run
#############################################################


# Imports
import itertools
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from build_df_raceData import*
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import csv
import pandas as pd
import numpy as np


df = build_df_years_processed(2018, 2023)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
#Sort by Driver and Race_ID
df = df.sort_values(['Race_ID', 'Driver','Race_Progress'])
#Save dataset
df.to_csv("dataset_2018_2023.csv")


#One hot encode Compound and track type
encoder = OneHotEncoder(sparse=False)
compound_encoded = encoder.fit_transform(df[['Relative_Compound']])
track_type_encoded = encoder.fit_transform(df[['Track_Type']])
track_status_encoded = encoder.fit_transform(df[['TrackStatus']])
stint_encoded = encoder.fit_transform(df[['Stint']])

# Convert boolean columns to integers
df['CloseBehind'] = df['CloseBehind'].astype(int)
df['CarClose'] = df['CarClose'].astype(int)
df['CloseAhead'] = df['CloseAhead'].astype(int)
df['InPits'] = df['InPits'].astype(int)
df['TwoCompoundsUsed'] = df['TwoCompoundsUsed'].astype(int)
df['PitstopBehind'] = df['PitstopBehind'].astype(int)



# Define predictor and target variables
features = [
    'CarClose','TwoCompoundsUsed', 'Race_Progress', 'TyreLife_Progress',
    'N_Driver_Champ_Pos', 'N_Position','Race_ID','PitstopBehind'
]

X = pd.concat([
    pd.DataFrame(compound_encoded).reset_index(drop=True),
    pd.DataFrame(track_type_encoded).reset_index(drop=True),
    pd.DataFrame(track_status_encoded).reset_index(drop=True),
    df[features].reset_index(drop=True)
], axis=1)

y = df[['InPits','Race_ID','Lap_ID','Driver']].copy()



# Split the dataset into training and testing sets
X_train=X[X['Race_ID']<=202300]#Select all races between 2018 and last race of 2022
y_train=y[y['Race_ID']<=202300]

X_val=X[X['Race_ID']>202300]#Select 2023 season
X_val=X_val[X_val['Race_ID']<202312]#Select first 10 races of 2023 as vaildation data

y_val=y[y['Race_ID']>202300]
y_val=y_val[y_val['Race_ID']<202312]

X_test=X[X['Race_ID']>=202312]#Select Last 10 races of 2023 as test data
y_test=y[y['Race_ID']>=202312]

X_test.to_csv('X_Test.csv')
y_test.to_csv('y_test.csv')



#Race ID used for dieving up test and training data, not required for training so can be dropped
X_train=X_train.drop(columns='Race_ID')
X_test=X_test.drop(columns='Race_ID')
X_val=X_val.drop(columns='Race_ID')

y_train=y_train.drop(columns=['Race_ID','Lap_ID','Driver'])
y_test=y_test.drop(columns=['Race_ID','Lap_ID','Driver'])
y_val=y_val.drop(columns=['Race_ID','Lap_ID','Driver'])


X_train=X_train.reset_index()
y_test=y_test.reset_index()
X_test=X_test.reset_index()
X_val=X_val.reset_index()
y_val=y_val.reset_index()
y_train.to_clipboard()
X_train = X_train.drop(columns=['index'])
X_test = X_test.drop(columns=['index'])
X_val=X_val.drop(columns=['index'])
y_val=y_val.drop(columns=['index'])
y_test = y_test.drop(columns=['index'])



# Define the hyperparameters based on the image provided
# l2_reg_values = [0.0003,0.0004,0.0005,0.0006]
# activation_functions = ['sigmoid','relu']
# neurons_layer_1 = [16,32,64]
# neurons_layer_2 = [16,32,64]
# neurons_layer_3 = [16,32,64]
# learning_rates = [1e-5,1e-4,1e-3]
# class_weights=[12,15,18,21]

l2_reg_values = [0.0003,0.00035,0.0004]
activation_functions = ['relu']
neurons_layer_1 = [16,32]
neurons_layer_2 = [64]
neurons_layer_3 = [16,32]
learning_rates = [1e-3]
class_weights=[20,21,22]
Uniq_combin=len(l2_reg_values)*len(activation_functions)*len(neurons_layer_1)*len(neurons_layer_2)*len(neurons_layer_3)*len(learning_rates)*len(class_weights)
print(f"There are {Uniq_combin} unique Combinations")
comb=1
# Function to create a model with given hyperparameters
def create_model(l2_reg, activation, neurons1, neurons2, neurons3, lr):
    # Define the model
    model = Sequential()
    model.add(Dense(neurons1,activation=activation,
                    kernel_regularizer=l2(l2_reg)))
    model.add(Dense(neurons2, activation=activation,
                    kernel_regularizer=l2(l2_reg)))
    model.add(Dense(neurons3, activation=activation,
                    kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function for hyperparameter tuning
def hyperparameter_tuning(x_train, y_train, x_val, y_val,x_test,y_test):
    results = []
    comb=1
    
    pred = pd.DataFrame()
    # Create all combinations of hyperparameters
    for l2_reg, activation, neurons1, neurons2, neurons3, lr, cw in itertools.product(
            l2_reg_values, activation_functions, neurons_layer_1, neurons_layer_2,
            neurons_layer_3, learning_rates,class_weights):
        print(f"Combination {comb} out of {Uniq_combin}")
        
        activation_layer = activation
        pd
        # Create a new model with the current set of hyperparameters
        model = create_model(l2_reg, activation_layer, neurons1, neurons2, neurons3, lr)
        class_weight = {0: 1, 1:cw}
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
        # Train the model using your data
        model.fit(x_train, y_train,
          class_weight=class_weight,
          batch_size=128,
          epochs=100,  
          validation_data=(x_val, y_val), 
          verbose=1,
          callbacks=[early_stopping])

        # Evaluate the model on the validation set
        loss, accuracy = model.evaluate(x_test, y_test)

        # Make predictions on the test set
        predictions = model.predict(x_test).flatten()
        pred[f"{comb}_pred"]=predictions
        # Store the results
        results.append({
            'Combination':comb,
            'L2 Regularization': l2_reg,
            'Activation Function': activation,
            'Layer 1 Neurons': neurons1,
            'Layer 2 Neurons': neurons2,
            'Layer 3 Neurons': neurons3,
            'Learning Rate': lr,
            'Class_Weight':cw,
            'Loss': loss,
            'Accuracy': accuracy
        })
        comb=comb+1
        # Write the results to a CSV file
        with open('hyperparameter_tuning_results.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
               'Combination' ,'L2 Regularization', 'Activation Function', 'Layer 1 Neurons',
                'Layer 2 Neurons', 'Layer 3 Neurons', 'Learning Rate','Class_Weight', 'Loss', 'Accuracy'
            ])
            writer.writeheader()
            for data in results:
                writer.writerow(data)

        pred.to_csv('Predictions.csv')


    return results

# Call function to run
results = hyperparameter_tuning(X_train, y_train, X_val, y_val,X_test,y_test)
