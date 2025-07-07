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
    


def save_model(model_directory: str, model_name:str, model:object):
    """
    Function to save the model.
    
    :param model_directory: Directory to save the model.
    :param model_name: Name of the model file.
    :param model: Model object.
    """
    try:
        os.makedirs(model_directory, exist_ok=True)
        model_path = os.path.join(model_directory, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print("Error in Saving Model File\t", e)
        raise e
    


def load_model(model_directory: str, model_name:str):
    """Function to load the model."""

    try:
        model_path = os.path.join(model_directory, f"{model_name}.pkl") 
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except Exception as e:
        print("Error in Loading Model File\t", e)
        return None



    
def main(file_path: str, target_column: str, processed_columns: list = None):
    """Main function to load data, train and evaluate the model."""


    df = load_dataset(file_path)
    df = df[processed_columns] if processed_columns else df
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    X, y = separate_features_and_target(df, target_column)
    
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)
    
    model = train_model(X_train, y_train)
    
    rmse, r2 = evaluate_model(model, X_test, y_test)
    print("Model Evaluation:")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Model Training Score: {model.score(X_train, y_train)}")
    print(f"RMSE: {rmse}")
    print(f"R^2 Score: {r2}")

    print("Saving Model: ")

    save_model(model_directory= "artifacts", model_name="linear_regression", model= model)




