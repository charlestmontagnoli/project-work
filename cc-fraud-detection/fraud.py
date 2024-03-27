#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime, date
import plotly.express as px

def clean_data():
    # import the data from the local file path
    file_path = "cc-fraud-detection\\cc_fraud_data\\cc_fraud_test.csv"
    df = pandas.read_csv(file_path)
    
    # let's check out the data a bit
    print("Raw Data:")
    print(df.columns)
    print(df.shape)
    # we can see that there are over 55k pieces of data, with 23 columns of information
    # now let's try to clean up the data
    df = df.drop('Unnamed: 0', axis = 1)
    # we're dropping the "unnamed" column because it doesn't seem to have any relevance
    #print("\nAfter Dropping Unnamed:")
    #print(df.columns)
    #print(df.shape)
    #viz_data(df)
    # I think there's more stuff to drop off as well:
    df = df.drop(['trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'],axis = 1)
    #print("\nAfter Dropping More Stuff:")
    #print(df.columns)
    #print(df.shape)
    # the most interesting piece of data that we have here is the fraudulent transactions,
    # so let's take a look at the fraudulent data
    print("\nFraud Count:")
    print(df['is_fraud'].value_counts())
    x = ['0','1']
    y = df['is_fraud'].value_counts()
    plt.bar(x,y)
    plt.xlabel('Non-Fraud and Fraud')
    plt.ylabel('Transaction Counts')
    plt.title('Instances of Fraudulent and Non-Fraudulent Transactions')
    plt.show()
    run_model(df)
    return

def viz_data(df):
    fraud_data = df[df.is_fraud == 1]
    fraud_data = fraud_data.drop(['first', 'last', 'street', 'cc_num', 'merchant', 'trans_num'],axis = 1)
    fraud_data['dob'] = pandas.to_datetime(fraud_data['dob'], dayfirst=True)
    c_day = datetime.now()
    fraud_data['age'] = (c_day-fraud_data['dob']).dt.days //365
    fraud_lat_lon = fraud_data.drop(['trans_date_trans_time', 'category', 'amt', 'gender', 'city', 'state',
       'zip', 'city_pop', 'job', 'dob', 'unix_time',
       'merch_lat', 'merch_long', 'is_fraud', 'age'],axis = 1)
    #print(fraud_lat_lon.columns)
    fig = px.scatter_mapbox(fraud_lat_lon, 
                        lat='lat',
                        lon='long')

    fig.update_layout(mapbox_style='open-street-map')

    fig.show()
    plt.bar(fraud_data['age'].unique(),fraud_data['age'].value_counts())
    plt.xlabel("Age")
    plt.ylabel("Instances of Fraud")
    plt.title("Fraud per Age Group")
    plt.show()
    plt.bar(fraud_data['gender'].unique(),fraud_data['gender'].value_counts())
    plt.xlabel("Gender")
    plt.ylabel("Instances of Fraud")
    plt.title("Fraud per Gender Group")
    plt.show()
    plt.bar(df['gender'].unique(),df['gender'].value_counts())
    plt.xlabel("Gender")
    plt.ylabel("Transactions")
    plt.title("Transaction per Gender Group")
    plt.show()
    return


def run_model(df):
    # let's try some learning
    # these categories need to be converted away from categorical variables
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit label encoder and return encoded labels
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['job'] = label_encoder.fit_transform(df['job'])
    df['dob'] = label_encoder.fit_transform(df['dob'])
    df['cc_num'] = label_encoder.fit_transform(df['cc_num'])
    df['merchant'] = label_encoder.fit_transform(df['merchant'])
    df['category'] = label_encoder.fit_transform(df['category'])

    x = df.drop('is_fraud', axis = 1) # x is everything but is_fraud
    y = df['is_fraud'] # y is only is_fraud
    # here we create the train/test/split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    # SMOTE implements the Synthetic Minority Over-Sampling strategy
    # this is used because our data is heavily imbalanced (55k vs 2.5k)
    smote = SMOTE(sampling_strategy = 'auto')
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    x_train = x_resampled
    y_train = y_resampled
    # here we setup a basic model
    tf.random.set_seed(123)
    model = Sequential([
        Dense(128, input_dim=x_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # binary crossentropy loss should be very effective for our data
    model.compile(optimizer = Adam(), 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
    # allow early stopping if it can
    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)
    # show the model summary
    model.summary()
    # run the model
    history = model.fit(x_train, y_train, epochs = 15, validation_split = 0.1, callbacks = [early_stopping])
    model.evaluate(x_test, y_test)
    #accuracy vs epochs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0.55,1.0)

    plt.legend(['training data', 'validation data'], loc = 'lower right')
    plt.show()

    #loss vs eopchs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0.000,0.50)

    plt.legend(['training data', 'validation data'], loc = 'upper right')
    plt.show()
    return


if __name__ == '__main__':
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    clean_data()