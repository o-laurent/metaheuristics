import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd


fig = plt.figure(figsize=(12, 12))
ax = fig.gca()

with open('dataframes_init_high_bit_stuck_cap.pickle', 'rb') as handle:
    dataframes = pickle.load(handle)

done = False
for df in dataframes:
    dico =  {i:df[0].iloc[i::25].values for i in range(25)}
    dico["level_0"] = df['level_0'].iloc[::25].values
    df = pd.DataFrame(dico).cummax().drop(columns=['level_0']).stack().reset_index().drop(columns='level_1')
    print(df)
    if not done:
        sns.lineplot(data=df, x="level_0", y=0, color="b", label="Capped acceptance probability")
    else:
        sns.lineplot(data=df, x="level_0", y=0, color="b")
    done = True

    

with open('dataframes_init_high_bit_stuck.pickle', 'rb') as handle:
    dataframes = pickle.load(handle)

done = False
for df in dataframes:
    dico =  {i:df[0].iloc[i::25].values for i in range(25)}
    dico["level_0"] = df['level_0'].iloc[::25].values
    df = pd.DataFrame(dico).cummax().drop(columns=['level_0']).stack().reset_index().drop(columns='level_1')
    print(df)
    if not done:
        sns.lineplot(data=df, x="level_0", y=0, color="r", ax=ax, label="Free probability")
    else:
        sns.lineplot(data=df, x="level_0", y=0, color="r")
    done = True

plt.title("Distribution of the score versus the number of the step")
plt.xlabel('Step')
plt.ylabel('Score')
plt.legend(loc='lower right')
fig.savefig("analysis/capped_notcapped.png")
plt.show()