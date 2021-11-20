import numpy as np
import matplotlib.pyplot as plt
from sho import make, algo, iters, bit, proba, temp
import optuna

# choice of the solver
solver = "num_simulated_annealing"
temperature = "proportional"

# parameters of the experiment
domain_width = 30
sensor_range = 0.3
d = 2
nb_sensors = 3
variation_scale = 0.3
target = 900  # continue until the end

# parameters of the analysis
mean_nb = 100
steps = 1500

history = []
iters = make.iter(
    iters.several,
    agains=[
        make.iter(iters.max,
                  nb_it=steps),
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


def median_results(temp_cst, cap, stuck_threshold):
    print(temp_cst, cap, stuck_threshold)
    val_array = []
    for _ in range(mean_nb):
        val, _ = algo.simulated_annealing(
            eval_function,
            init_function,
            neigh_function,
            iters,
            make.proba(proba.exponential),
            make.temp(temperature_func, max_iter=temp_cst),
            cap,
            stuck_threshold
        )
        val_array.append(val)
        history.clear()
    return np.mean(val_array)


def optimize(trial):
    """Wrapper for optuna"""
    temp_cst = trial.suggest_uniform('temp_cst', 0, 10000)
    cap = trial.suggest_uniform('cap', 0, 1)
    stuck_threshold = trial.suggest_int('stuck', 50, 800)
    return -median_results(temp_cst, cap, stuck_threshold)


optuna.logging.set_verbosity(optuna.logging.DEBUG)
study = optuna.create_study()
study.optimize(optimize, n_jobs=1, n_trials=200)
print(study.best_params)

{'temp_cst': 2, 'cap': 0.13815523842909097, 'stuck': 737}
