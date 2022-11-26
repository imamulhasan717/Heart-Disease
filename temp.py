import numpy as np
import pandas as pd
import matplotlib.pyplot as splt
import seaborn
df=pd.read_csv('cardio_train.csv',sep=';')
df.shape
df['cardio'].value_counts()
seaborn.countplot(df['cardio'])
df.isnull().values.any()
df.isna().sum()
seaborn.countplot(x='gender',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
seaborn.countplot(x='age',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
df['year']=(df['age']/365).round(0)
seaborn.countplot(x='year',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
df.describe()
df.corr()
df=df.drop(['year'],axis=1)
df=df.drop(['id'],axis=1)
x=df.iloc[:,:-1]
y=df.iloc[:,12]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)
from sklearn.ensemble import RandomForestClassifier
Rclf=RandomForestClassifier()
Rclf.fit(xtrain,ytrain)
Rclf.score(xtest,ytest)
from sklearn.tree import DecisionTreeClassifier
Clf=DecisionTreeClassifier()
Clf.fit(xtrain, ytrain)
Clf.score(xtest,ytest)
