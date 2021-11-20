import numpy as np
import matplotlib.pyplot as plt
from sho import make, algo, iters, plot, num, bit, pb, proba, temp
import seaborn as sns
import pandas as pd
import pickle

# choice of the solver
solver = "bit_simulated_annealing"
temperature = "proportional"

# parameters of the experiment
domain_width = 30
sensor_range = 0.3
d = 2
nb_sensors = 3
variation_scale = 0.3
target = 900  # continue until the end

# parameters of the analysis
mean_nb = 25
steps = 1500

history = []
iters = make.iter(
    iters.several,
    agains=[
        make.iter(iters.max,
                  nb_it=steps),
        make.iter(iters.save,
                  filename=solver+".csv",
                  fmt="{it} ; {val} ; {sol}\n"),
        make.iter(iters.log,
                  fmt="\r{it} {val}"),
        make.iter(iters.history,
                  history=history),
        make.iter(iters.target,
                  target=target),
        # iters.max(steps)
    ]
)

eval_function = make.func(bit.cover_sum,
                          domain_width=domain_width,
                          sensor_range=sensor_range,
                          dim=d * nb_sensors
                          )
init_function = make.init(bit.rand_init_halt,
                          domain_width=domain_width,
                          nb_sensors=nb_sensors)
neigh_function = make.neig(bit.neighb_square,
                           scale=variation_scale,
                           domain_width=domain_width)
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()

if temperature == "root":
    temperature_func = temp.root
elif temperature == "proportional":
    temperature_func = temp.proportional
elif temperature == "log":
    temperature_func = temp.logarithmic
    
dataframes = []
for stuck in range(100, 701, 100):
    whole_history = []
    for _ in range(mean_nb):
        val, sol = algo.simulated_annealing(
            eval_function,
            init_function,
            neigh_function,
            iters,
            make.proba(proba.exponential),
            make.temp(temperature_func, max_iter=800),
            stuck=stuck
        )
        whole_history.append(np.array(list(map(lambda x: x[0], history))))
        history.clear()
    sensors = bit.to_sensors(sol)
    whole_history = np.array(whole_history).T #transpose
    df = pd.DataFrame(whole_history, columns=[
                          'exp '+str(i) for i in range(1, mean_nb+1)]).stack().reset_index().drop(columns='level_1')
    dataframes.append(df)
    # sns.lineplot(data=df, x="level_0", y=0)

with open('dataframes_init_high_bit_varstuck.pickle', 'wb') as handle:
    pickle.dump(dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plt.show()
