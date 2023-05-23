import pandas as pd
from matplotlib import pyplot as plt


def generate_mean_training_return_plot(sb3_log_paths):
    fig, ax = plt.subplots()
    for sb3_log_path in sb3_log_paths:
        add_mean_training_return_of_run(sb3_log_path, ax)
    return fig


def add_mean_training_return_of_run(sb3_log_path, ax):
    mean_training_return = load_mean_training_return(sb3_log_path)
    mean_training_return.dropna(inplace=True)  # drop rows that correspond to evaluation logs
    ax.plot(mean_training_return["time/total_timesteps"], mean_training_return["rollout/ep_rew_mean"])


def load_mean_training_return(sb3_log_path):
    mean_training_return = \
        pd.read_csv(sb3_log_path + "/" + "progress.csv", usecols=["time/total_timesteps", "rollout/ep_rew_mean"])
    return mean_training_return
