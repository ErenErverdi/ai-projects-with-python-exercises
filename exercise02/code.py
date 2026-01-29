import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('multilinearregression.csv', sep=';')

x = df[["alan","odasayisi","binayasi",]]
y = df["fiyat"]

rg = linear_model.LinearRegression()
rg.fit(x,y)

rg.predict([[230,4,10]])
rg.predict([[230,6,0]])
rg.predict([[365,3,20]])
rg.coef_
rg.intercept_

# Model Handmade 
a = rg.intercept_
b1 = rg.coef_[0]
b2 = rg.coef_[1]
b3 = rg.coef_[2]

x1 = 230
x2 = 4
x3 = 10

y = a + b1*x1 + b2*x2 + b3*x3
y
