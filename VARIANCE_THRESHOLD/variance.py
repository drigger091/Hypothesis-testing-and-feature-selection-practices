import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

df =pd.read_csv("G:\Data Science(Hypothesis testing and feature selection)\VARIANCE_THRESHOLD\santander.csv",nrows=7000)

X = df.drop(columns="TARGET",axis=1)
Y = df["TARGET"]

# using train test split

X_TRAIN , X_TEST ,Y_TRAIN , Y_TEST = train_test_split(X,Y,test_size=0.4,random_state=0)

var_threshold = VarianceThreshold(threshold=0)
var_threshold.fit(X_TRAIN)

#print(var_threshold.get_support())

#print(len(X_TRAIN.columns[var_threshold.get_support()]))

constant_columns = [column for column in X_TEST.columns
                    if column not in X_TRAIN.columns[var_threshold.get_support()]]

print(len(constant_columns))

X_TRAIN = X_TRAIN.drop(constant_columns,axis=1)
print(X_TRAIN.head())