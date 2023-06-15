import os
import matplotlib.pyplot as plt

from models.rainbow_model import rainbow_model
from models.dqn_model import dqn_model
from models.random_model import random_model


def get_number_graphs(dir_path):
    number_graphs = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            number_graphs += 1
    return number_graphs


class controller:

    def __init__(self):
        self.rainbow_model = rainbow_model()
        self.rainbow_rewards = []
        self.rainbow_times = []

        self.dqn_model = dqn_model()
        self.dqn_rewards = []
        self.dqn_times = []

        self.random_model = random_model()
        self.random_rewards = []
        self.random_times = []

        self.num_iterations = None
        self.learning_timesteps = None

    def set_graph_name_variables(self, num_iterations, learning_timesteps):
        self.num_iterations = num_iterations
        self.learning_timesteps = learning_timesteps

    # Runs the Rainbow Agent
    def run_rainbow(self, num_iterations):
        self.rainbow_model.set_num_iterations(num_iterations)
        self.rainbow_model.run_agent()
        self.rainbow_rewards = self.rainbow_model.rewards
        self.rainbow_times = self.rainbow_model.times

    # Runs the DQN Agent
    def run_dqn(self, num_iterations, learning_timesteps):
        self.dqn_model.set_num_iterations(num_iterations)
        self.dqn_model.set_learning_timesteps(learning_timesteps)
        self.dqn_model.run_agent()
        self.dqn_rewards = self.dqn_model.rewards
        self.dqn_times = self.dqn_model.times

    # Runs the random agent
    def run_random(self, num_iterations):
        self.random_model.set_num_iterations(num_iterations)
        self.random_model.run_agent()
        self.random_rewards = self.random_model.rewards
        self.random_times = self.random_model.times

    def run_all(self, num_iterations, learning_timesteps):
        self.num_iterations = num_iterations
        self.learning_timesteps = learning_timesteps
        self.run_rainbow(num_iterations)
        self.run_dqn(num_iterations, learning_timesteps)
        self.run_random(num_iterations)

    # Plots Avg Rewards from all 3 agents onto a matplotlib graph
    def plot_rewards(self):
        number_graphs = get_number_graphs(r'graphs/rewards')
        plt.plot(self.rainbow_model.steps, self.rainbow_rewards, label="Rainbow")
        plt.plot(self.dqn_model.steps, self.dqn_rewards, label="DQN")
        plt.plot(self.random_model.steps, self.random_rewards, label="Random")
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.title('Avg Reward vs Step')
        plt.ylim(top=550)
        plt.grid(axis='x', color='0.95')
        plt.legend()
        plt.savefig(f'graphs/rewards/Avg Rewards Comparison {number_graphs + 1} - {self.num_iterations} Iterations - {self.learning_timesteps} Timesteps.jpg')
        plt.close()

    # Plots Avg Times from all 3 agents onto a matplotlib graph
    def plot_times(self):
        number_graphs = get_number_graphs(r'graphs/times')
        plt.plot(self.dqn_model.steps, self.dqn_times, label="DQN")
        plt.plot(self.random_model.steps, self.random_times, label="Random")
        plt.xlabel('Step')
        plt.ylabel('Average Time')
        plt.title('Avg Time vs Step')
        plt.ylim(top=1)
        plt.grid(axis='x', color='0.95')
        plt.legend()
        plt.savefig(f'graphs/times/Avg Times Comparison {number_graphs + 1} - {self.num_iterations} Iterations - {self.learning_timesteps} Timesteps.jpg')
        plt.close()

    # Outputs steps and rewards for each agent, used for testing purposes
    def output_variables(self):
        print(f'Rainbow Steps: {self.rainbow_model.steps}')
        print('-------------------------------------------')
        print(f'DQN Steps: {self.dqn_model.steps}')
        print('-------------------------------------------')
        print(f'Rainbow Rewards: {self.rainbow_rewards}')
        print('-------------------------------------------')
        print(f'DQN Rewards: {self.dqn_rewards}')
        print('-------------------------------------------')
        print(f'Random Steps: {self.random_model.steps}')
        print('-------------------------------------------')
        print(f'Random Rewards: {self.random_rewards}')
