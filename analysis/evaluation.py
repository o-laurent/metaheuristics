import numpy as np
import matplotlib.pyplot as plt
from sho import make, algo, iters, plot, num, bit, pb, proba, temp
import seaborn as sns
import pandas as pd

def to_heatmap(arr):
    heatmap = np.zeros((1501, 685))
    for i, val in enumerate(arr):
        for j in range(val+1):
            heatmap[i, j] = 1
    return heatmap[:-1, :].T

# choice of the solver
solver = "bit_simulated_annealing"
temperature = "proportional"

# choice of the plot
plot_type = "heatmap"  # seaborn or heatmap for ECDF
evaluation = "ECDF"  # "EAF_Slice" or "ECDF"
save = True

# parameters of the experiment
domain_width = 30
sensor_range = 0.3
d = 2
nb_sensors = 3
variation_scale = 0.3
target = 900  # continue until the end

# parameters of the analysis
mean_nb = 200
steps = 1500

history = []
iters = make.iter(
    iters.several,
    agains=[
        make.iter(iters.max,
                  nb_it=steps),
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
init_function = make.init(bit.rand,
                          domain_width=domain_width,
                          nb_sensors=nb_sensors)
neigh_function = make.neig(bit.neighb_square,
                           scale=variation_scale,
                           domain_width=domain_width)
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()

if evaluation == "EAF_Slice":
    if solver == "bit_simulated_annealing":
        if temperature == "root":
            temperature_func = temp.root
        elif temperature == "proportional":
            temperature_func = temp.proportional
        whole_history = []
        for i in range(mean_nb):
            val, sol = algo.simulated_annealing(
                eval_function,
                init_function,
                neigh_function,
                iters,
                make.proba(proba.exponential),
                make.temp(temperature_func, max_iter=2),
                cap = 0.13815,
                stuck = 737
            )
            whole_history.append(np.array(list(map(lambda x: x[0], history))))
            history.clear()
        sensors = bit.to_sensors(sol)
        whole_history = np.array(whole_history)

    for delta in range(500, 651, 50):
        vals = []
        for i in range(10, steps):
            vals.append(np.mean(whole_history[:, i] > delta))
        line, = ax.plot(list(range(10, steps)), vals)
        line.set_label("Threshold: "+str(int(delta)))
    ax.set_xscale('log')
    plt.title("Slices of the Empirical Attainment Function")
    plt.xlabel('step')
    plt.ylabel('% of values above threshold')
    plt.legend(loc='lower right')
    plt.show()
    if save:
        fig.savefig("analysis/EAF_Slice.png")
elif evaluation == "ECDF":
    if solver == "bit_simulated_annealing":
        if temperature == "root":
            temperature_func = temp.root
        elif temperature == "proportional":
            temperature_func = temp.proportional
        whole_history = []
        for i in range(mean_nb):
            val, sol = algo.simulated_annealing(
                eval_function,
                init_function,
                neigh_function,
                iters,
                make.proba(proba.exponential),
                make.temp(temperature_func, max_iter=steps),
            )
            part_history = np.array(list(map(lambda x: x[0], history))).astype(int)
            whole_history.append(to_heatmap(part_history))
            history.clear()
        sensors = bit.to_sensors(sol)
        whole_history = np.array(whole_history)

    if plot_type == "heatmap":
        vals = []
        # for i in range(10, steps):
        #     vals.append(np.mean(whole_history[:, :i]))
        # plt.plot(list(range(10, steps)), vals)
        sns.heatmap(np.mean(whole_history, axis=0))

    elif plot_type == "seaborn":
        sns.set_theme(style="whitegrid")
        df = pd.DataFrame(whole_history.T, columns=[
                          'exp '+str(i) for i in range(1, mean_nb+1)])
        scatterplot = sns.scatterplot(data=df, alpha=0.05, s=30, markers=["o"]*mean_nb, palette=['b']*mean_nb, ax=ax)
        plt.legend([],[], frameon=False)

    plt.title("Empirical Cumulative Distribution Function")
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.show()
    if save:
        fig.savefig("analysis/ECDF_"+plot_type+".png")

