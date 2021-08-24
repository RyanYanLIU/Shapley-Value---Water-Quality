import pandas as pd 
import numpy as np 

df = pd.read_csv('water_potability.csv')
print(f'Total Length of dataframe is: {len(df)}')
def fill_nan(data):
    data = df
    for col in data.columns:
        Avg = data[col].mean()
        data.fillna(Avg, inplace = True)
    return data 

def remove_outliers(data):
    data = df 
    for col in data.columns:
        Avg, Std = data[col].mean(), data[col].std()
        Low, High = Avg - 3 * Std, Avg + 3 * Std
        data = data[data[col].between(Low, High, inclusive = True)]
        return data

'''X = list(map(fill_nan, df))[0]
X1 = list(map(remove_outliers, X))[0]'''
my_df = df.copy()
data_pipe = (my_df.pipe(fill_nan).pipe(remove_outliers))

print(data_pipe)