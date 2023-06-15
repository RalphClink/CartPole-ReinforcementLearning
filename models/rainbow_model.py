from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.timer import timer

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


def compute_avg_reward(environment, policy, num_episodes=10):
    total_reward = 0.0

    for i in range(num_episodes):
        time_step = environment.reset()
        episode_reward = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward

        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    return avg_reward.numpy()[0]


class rainbow_model:

    def __init__(self):
        self.env_name = "CartPole-v1"
        self.num_iterations = 15000

        self.initial_collect_steps = 1000
        self.collect_steps_per_iteration = 1
        self.replay_buffer_capacity = 10000
        self.fc_layer_params = (100,)
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.log_interval = 200
        self.num_atoms = 51
        self.min_q_value = -20
        self.max_q_value = 20
        self.n_step_update = 2

        self.num_eval_episodes = 10
        self.eval_interval = 1000

        # Load environments, one for training, one for evaluation
        self.train_py_env = suite_gym.load(self.env_name)
        self.eval_py_env = suite_gym.load(self.env_name)
        # Convert environments to TensorFlow
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.train_py_env)

        # Agent
        self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            num_atoms=self.num_atoms,
            fc_layer_params=self.fc_layer_params)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = categorical_dqn_agent.CategoricalDqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            categorical_q_network=self.categorical_q_net,
            optimizer=self.optimizer,
            min_q_value=self.min_q_value,
            max_q_value=self.max_q_value,
            n_step_update=self.n_step_update,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=self.gamma,
            train_step_counter=self.train_step_counter)
        self.agent.initialize()

        # Replay Buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_capacity)

        # Random Policy
        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                             self.train_env.action_spec())
        compute_avg_reward(self.eval_env, self.random_policy, self.num_eval_episodes)

        # Dataset
        self.dataset = None
        self.iterator = None

        # Variables to store performance info
        self.rewards = []
        self.times = []

        # Plotting Variable
        self.steps = None

    def collect_step(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)
        # Set up the dataset and iterator now that the replay buffer isn't empty
        if self.dataset is None:
            self.dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=3, sample_batch_size=self.batch_size,
                num_steps=self.n_step_update + 1).prefetch(3)
            self.iterator = iter(self.dataset)

    def collect_step_loop(self):
        for x in range(self.initial_collect_steps):
            self.collect_step(self.train_env, self.random_policy)

    def set_plot_variables(self):
        self.steps = range(0, self.num_iterations + 1, self.eval_interval)

    def set_num_iterations(self, number_iterations):
        self.num_iterations = number_iterations

    def train_agent(self):
        # Optimize by wrapping some code in a graph using TensorFlow
        self.agent.train = common.function(self.agent.train)

        # Reset the training step
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training
        avg_reward = compute_avg_reward(self.eval_env, self.agent.policy, self.num_eval_episodes)
        self.rewards = [avg_reward]

        for y in range(self.num_iterations):
            # Collect a few steps using collect_policy, save to replay buffer
            for z in range(self.collect_steps_per_iteration):
                self.collect_step(self.train_env, self.agent.collect_policy)

            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience)

            step = self.agent.train_step_counter.numpy()
            # print(self.agent.train_step_counter)

            if step % self.log_interval == 0:
                print(f'step = {step}: loss = {train_loss.loss}')

            if step % self.eval_interval == 0:
                avg_reward = compute_avg_reward(self.eval_env, self.agent.policy, self.num_eval_episodes)
                print(f'step = {step}: Average reward = {avg_reward:.2f}')
                self.rewards.append(avg_reward)

    # Runs the agent, this is the only method you need to
    # run once you've created an instance of the class
    def run_agent(self):
        self.collect_step_loop()
        self.train_agent()
        self.set_plot_variables()
