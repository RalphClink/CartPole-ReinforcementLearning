import gym

from statistics import mean
from stable_baselines3 import DQN
from tf_agents.environments import suite_gym
from models.timer import timer

class dqn_model:

    def __init__(self):
        self.env_name = "CartPole-v1"
        self.env = gym.make(self.env_name)
        self.num_iterations = 15000
        self.log_interval = 200
        self.num_eval_episodes = 10
        self.eval_interval = 1000
        self.steps = None

        # Hyperparamaters (can be added to model = DQN())
        self.learning_timesteps = 100000
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.batch_size = 512
        self.tau = 0.02

        # Variables to store performance info
        self.rewards = []
        self.times = []
        self.timer = timer()

    def set_learning_timesteps(self, learning_timesteps):
        self.learning_timesteps = learning_timesteps

    def set_num_iterations(self, num_iterations):
        self.num_iterations = num_iterations

    def set_steps(self):
        self.steps = range(0, self.num_iterations + 1, self.eval_interval)

    def run_agent(self):
        temp_rewards = []
        avg_rewards = []
        temp_times = []

        self.env.reset()

        model = DQN("MlpPolicy", env=self.env, verbose=1)

        model.learn(total_timesteps=self.learning_timesteps, log_interval=self.eval_interval)
        obs = self.env.reset()

        episode_number = 0
        new_attempt = True

        while episode_number <= self.num_iterations:
            if new_attempt:
                self.timer.start()
                new_attempt = False
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            temp_rewards.append(reward)
            if done:
                avg_rewards.append(sum(temp_rewards))
                temp_times.append(self.timer.stop())

                # Add the average reward from the evaluation interval to the master reward list
                if episode_number % self.eval_interval == 0:
                    self.rewards.append(mean(avg_rewards))
                    self.times.append(mean(temp_times))

                    temp_times = []
                    avg_rewards = []

                temp_rewards = []
                episode_number += 1
                new_attempt = True
                obs = self.env.reset()

        self.set_steps()


