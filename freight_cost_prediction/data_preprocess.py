import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_data(db_path:str):
    conn=sqlite3.connect(db_path)
    query="select * from vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df


def feature_extraction(df:pd.DataFrame):
    X=df[['Dollars','Quantity']]
    Y=df['Freight']
    return X,Y

def data_split(X,Y):
    return train_test_split(X,Y,test_size=0.2,random_state=42)



