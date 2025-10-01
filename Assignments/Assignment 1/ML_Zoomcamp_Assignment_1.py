# %%
import pandas as pd
import numpy as np

# %%
pd.__version__

# %%
data = pd.read_csv("Assignments/Assignment 1/car_fuel_efficiency.csv")

data

# %%
data.info()

# %%
data.fuel_type.unique()

# %%
data.isna().sum()

# %%
data[data['origin'] == 'Asia'].groupby('fuel_efficiency_mpg').max()

# %%
print(data.horsepower.median())

# %%
data.horsepower.value_counts()

# %%
print(data.horsepower.mode())

# %%
data['horsepower'] = data['horsepower'].fillna(152.0)

# %%
print(data.isna().sum())

# %%
print(data.horsepower.median())

# %%
X = data.loc[data['origin'] == 'Asia', ['vehicle_weight', 'model_year']].head(7).values

print(X)

# %%
Y = X.T

print(Y)

# %%
XTX = Y.dot(X)

print(XTX)

# %%
XTX_inv = np.linalg.inv(XTX)

print(XTX_inv)

# %%
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

print(y)

# %%
C = XTX_inv.dot(Y)

print(C)

# %%
w = C.dot(y)

print(w)

# %%
print(w.sum())



