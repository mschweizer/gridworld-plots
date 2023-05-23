import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

GRID = np.array([[2, 2, 2, 2, 2, 2],
                 [2, 9, 9, 9, 9, 2],
                 [2, 1, 1, 1, 8, 2],
                 [2, 1, 1, 1, 1, 2],
                 [2, 1, 1, 1, 1, 2],
                 [2, 2, 2, 2, 2, 2]])

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
UNDEFINED = -1

ARROW_LENGTH = 0.4


def add_arrows(policy_map, ax):
    for (x, y), action in np.ndenumerate(policy_map):
        if action in (UP, DOWN, LEFT, RIGHT):
            if action == RIGHT:
                dx, dy = ARROW_LENGTH, 0
            elif action == DOWN:
                dx, dy = 0, ARROW_LENGTH
            elif action == LEFT:
                dx, dy = -ARROW_LENGTH, 0
            else:
                dx, dy = 0, -ARROW_LENGTH

            ax.arrow(x, y, dx, dy, head_width=0.1, length_includes_head=True, color="red")


def generate_policy_map(agent):
    base_obs = generate_base_obs()

    policy_map = np.full(shape=(6, 6), fill_value=-1.)
    for x in range(1, 5):
        for y in range(2, 5):
            obs = base_obs.copy()
            if obs[x, y, 0] != 8:
                obs[x, y, 0] = 10
                # print("agent position (x,y)")
                # print(str(x) + str(y))
                # print("Obs for prediction")
                # print(obs.transpose())
                actions, _ = agent.predict(obs, deterministic=True)
                action = actions[()]
                policy_map[x, y] = action
    return policy_map


def generate_base_obs():
    agent_orientation = np.zeros((6, 6))
    base_obs = np.array([GRID, agent_orientation]).transpose()
    return base_obs


def generate_policy_map_plot(sb3_log_path=None, agent=None):
    if not agent:
        agent = DQN.load(sb3_log_path + "/final_model.zip")

    fig, ax = plt.subplots()
    displayed_grid = GRID.copy()
    displayed_grid[2, 1] = 4
    ax.imshow(displayed_grid)

    policy_map = generate_policy_map(agent)
    add_arrows(policy_map, ax)

    return fig


if __name__ == '__main__':
    generate_policy_map_plot("")
