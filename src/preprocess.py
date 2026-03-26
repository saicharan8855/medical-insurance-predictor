import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv('data/insurance.csv')
    
    le = LabelEncoder()
    df['smoker'] = le.fit_transform(df['smoker'])
    df['sex'] = le.fit_transform(df['sex'])
    df['region'] = le.fit_transform(df['region'])
    
    X = df.drop(columns='charges')
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler