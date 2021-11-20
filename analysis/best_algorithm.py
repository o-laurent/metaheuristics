import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()

with open('dataframes_init_high_bit_stuck.pickle', 'rb') as handle:
    dataframes = pickle.load(handle)

for i, df in enumerate(dataframes):
    dico = {i: df[0].iloc[i::25].values for i in range(25)}
    dico["level_0"] = df['level_0'].iloc[::25].values
    df = pd.DataFrame(dico).cummax().drop(columns=['level_0'])
    print(i, df.median(axis=1)[1000], df.mean(
        axis=1)[1000], df.var(axis=1)[1000])

df = dataframes[9]
dico = {i: df[0].iloc[i::25].values for i in range(25)}
dico["level_0"] = df['level_0'].iloc[::25].values
df = pd.DataFrame(dico).cummax().drop(
    columns=['level_0']).stack().reset_index().drop(columns='level_1')
print(df)

sns.lineplot(data=df, x="level_0", y=0, color="r",
             ax=ax, label="Best solution")
plt.show()
