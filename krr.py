import numpy
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import os
import sys
import pandas
from sklearn.metrics import mean_absolute_percentage_error

path = "/home/nata/pythonProj/STRAPS/"

df = pd.read_csv(path+"results.txt", header=None)

print(df)

# learn_df = df.loc[0:1499]
learn_df = df

x = learn_df[learn_df.columns[2:]].to_numpy()

y = learn_df[1].values

krr = KernelRidge(alpha=1.0)
krr.fit(x, y)


import pickle

# save
with open('model.pkl','wb') as f:
    pickle.dump(krr,f)

# load


# check_df = df.loc[1500:]
new_df = pd.read_csv(path+"melissa_for_krr.txt", header=None)
# check_df = df.loc[1500:]
check_df = new_df
x_to_test = check_df[check_df.columns[2:]].to_numpy()
print(x_to_test.shape)
print(x_to_test[0].shape)
y_to_test = check_df[1].values


with open('model.pkl', 'rb') as f:
    krr2 = pickle.load(f)



res = krr2.predict(x_to_test)
print(type(res))
print(res[0:15])
print(y_to_test[0:15])
# print("correct value = ", y_to_test[0])



mape = mean_absolute_percentage_error(y_to_test, res)
print(mape)

# n_samples, n_features = 10, 5
# rng = np.random.RandomState(0)
# y = rng.randn(n_samples)
# X = rng.randn(n_samples, n_features)
# krr = KernelRidge(alpha=1.0)
# krr.fit(X, y)
# KernelRidge(alpha=1.0)
