import os

from plots.mean_training_return import generate_mean_training_return_plot
from plots.policy_map import generate_policy_map_plot

if __name__ == '__main__':
    os.makedirs("results/plots", exist_ok=True)

    run_1_log_path = "results/sb3/env=BeachWalk-v0_algo=dqn_nsteps=60000_7_energetic-cougar"
    run_2_log_path = "results/sb3/env=BeachWalk-v0_algo=dqn_nsteps=100000_6_quick-coyote"

    fig = generate_mean_training_return_plot([run_1_log_path, run_2_log_path])
    fig.show()
    fig.savefig("results/plots/mean_training_return.pdf")

    fig = generate_policy_map_plot(run_1_log_path)
    fig.show()
    fig.savefig("results/plots/policy_map_1.pdf")

    fig = generate_policy_map_plot(run_2_log_path)
    fig.show()
    fig.savefig("results/plots/policy_map_2.pdf")


