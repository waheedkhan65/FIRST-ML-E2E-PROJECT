import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

def load_dataset(filepath="data\processed\train.csv"):
    ''' Load dataset and also return it as a pandas DataFrame'''
    
    return pd.read_csv(filepath)


def separat_features_target(df, target_column):
    '''sepearet features and target columns from the dataset'''

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X,y


def split_data(X,y, test_size=0.2, random_state=42):
    '''Split the dataset into training and testing sets'''
    
    return train_test_split(X, y, test_size=0.2, randome_state=42)


def train_model(X_train, y_train):
    '''Train the model'''
    model = LinearRegression()
    y_pred = model.fit(X_train, y_train)
    return model

def Evaluate_model(model, X_test, y_test):

    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 =  r2_score(y_test, prediction)
    

def save_model():
    pass

def main():
    pass





