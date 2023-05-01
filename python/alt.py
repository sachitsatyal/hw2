
import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(
            input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(
            input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(
            params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(
            params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = list()
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(
                policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            sum_of_rewards = 0
            reward_list = self.trajectory.get('reward')
            for trajectory_reward_list in reward_list:
                sum_of_rewards += apply_return(trajectory_reward_list)
            avg_ro_reward = (sum_of_rewards/len(reward_list)).item()
            print(
                f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)

        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video(5000)
        # Close environment
        self.env.close()

    def update_critic_net(self):
        torch.autograd.set_detect_anomaly(True)
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                
                
                # I think we need to update state values in each epoch because critic_net is being updated every time 
                # and we need to calculate state values using updated critic_net. Otherwise, in a particular iteration, 
                # for n epochs, neither state values change nor target values change 
                # and so we loss value does not change which implies no meaning for multiple epochs we are performing
                self.update_state_values()
                critic_loss = self.estimate_critic_loss_function()
                # critic_loss_copy = critic_loss.clone()
                # critic_loss_copy.to(torch.float)
                # print('Iteration:',critic_iter_idx,' Epoch:', critic_epoch_idx, critic_loss)
                # print('critic_loss.shape:', critic_loss.shape)
                # print('critic_loss_copy.shape:', critic_loss_copy.shape)
                # print('critic_loss.version:', critic_loss._version)
                # print('critic_loss_copy.version:', critic_loss_copy._version)
                critic_loss.backward(retain_graph = True)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

    def update_target_value(self, gamma=0.99):
        # TODO: Update target values
        # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF

        trajectory_observations = self.trajectory['obs']
        trajectory_rewards = self.trajectory['reward']
        trajectory_target_values = []

        for trajectory_idx in range(len(trajectory_observations)):
            trajectory_rewards_idx = trajectory_rewards[trajectory_idx]
            rewards_tensor = torch.tensor(trajectory_rewards_idx, dtype=torch.float32).unsqueeze(1).detach()
            obs_tensor = trajectory_observations[trajectory_idx]
            next_obs_tensor = torch.cat([obs_tensor[1:].clone(), torch.zeros(1, obs_tensor.shape[1])], dim=0)
            next_state_values_tensor = self.critic_net(next_obs_tensor)
            target_values_tensor = rewards_tensor + gamma * next_state_values_tensor.detach()
            trajectory_target_values.append(target_values_tensor)

        self.trajectory['target_value'] = trajectory_target_values

    def update_target_value(self, gamma=0.99):
        # Retrieve trajectory data
        observations = self.trajectory['obs']
        rewards = self.trajectory['reward']

        # Compute target values for each time step in the trajectory
        target_values = []
        for t in range(len(observations)):
            # Compute discounted future reward
            future_reward = torch.tensor(0.0)
            discount = 1
            for k in range(t, len(observations)):
                future_reward += discount * torch.tensor(rewards[k], dtype=torch.float32)
                discount *= gamma

            # Compute target value using Bellman equation
            target_value = future_reward
            if t < len(observations) - 1:
                next_observation = observations[t + 1]
                if next_observation.shape[0] != observations[t].shape[0]:
                    # Pad the next observation with zeros if it has a different time dimension than the current observation
                    next_observation = torch.cat([next_observation, torch.zeros(observations[t].shape[0] - next_observation.shape[0], next_observation.shape[1])])
                next_state_value = self.critic_net(next_observation)
                target_value += gamma * next_state_value.detach()

            target_values.append(target_value)

        # Store target values in trajectory dictionary
        self.trajectory['target_value'] = target_values



    def update_state_values(self):
        trajectory_observations = self.trajectory['obs']
        trajectory_state_values = []

        for trajectory_idx in range(len(trajectory_observations)):
            trajectory_observations_idx = trajectory_observations[trajectory_idx]
            #obs_tensor = torch.tensor(trajectory_observations_idx, dtype=torch.float32)
            state_values_tensor = self.critic_net(trajectory_observations_idx)
            trajectory_state_values.append(state_values_tensor)
            #print(trajectory_observations_idx)
            #print(obs_tensor)
            #print(state_values_tensor)
            
        # print(trajectory_state_values)
        # print(0/0)
        self.trajectory['state_value'] = trajectory_state_values
            
    
    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF
        #self.trajectory['advantage'] = [a - b for a, b in zip(self.trajectory['target_value'], self.trajectory['state_value'])]
        #Fetching the upated values from the Critic Model with updated weights
        self.update_state_values()
        self.update_target_value()
        state_values = self.trajectory['state_value']
        target_values = self.trajectory['target_value']
        advantage_values = list()
        for trajectory_idx in range(len(state_values)):
            state_values_idx = state_values[trajectory_idx]
            target_values_idx = target_values[trajectory_idx]
            advantage_values_idx = list()
            for i in range(len(state_values_idx)):
                adv = target_values_idx[i] - state_values_idx[i]
                advantage_values_idx.append(adv.detach())
            advantage_values.append(advantage_values_idx)
        self.trajectory['advantage'] = advantage_values

    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def estimate_critic_loss_function(self):
        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.

        state_values = self.trajectory['state_value']
        target_values = self.trajectory['target_value']
        fn = torch.nn.MSELoss()
        critic_loss = torch.tensor(0.0, dtype=torch.float32, device=get_device())
        for trajectory_idx in range(len(state_values)):
            state_values_idx = state_values[trajectory_idx]
            target_values_idx = target_values[trajectory_idx]
            # state_values_tensor = torch.stack(state_values_idx).squeeze()  # Convert state_values_idx to a PyTorch tensor and detach it
            # target_values_tensor = torch.stack(target_values_idx).squeeze()  # Convert target_values_idx to a PyTorch tensor
            # print(state_values_tensor)
            # print(target_values_tensor)
            # print(state_values_idx)
            # print(target_values_idx)
            #print(0/0)
            critic_loss = critic_loss + fn(state_values_idx, target_values_idx)
        #print(critic_loss)
        return critic_loss

    def estimate_actor_loss_function(self):
        actor_loss = list()
        log_probabilities_list = self.trajectory.get('log_prob')
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            log_probabilities_list_idx = log_probabilities_list[t_idx]
            # print(self.trajectory['advantage'][t_idx][0])
            # print('#################')
            advantage = apply_discount(self.trajectory['advantage'][t_idx])
            # TODO: Compute actor loss function
            loss_idx = 0
            for idx in range(len(log_probabilities_list_idx)):
                loss_idx = loss_idx + \
                    log_probabilities_list_idx[idx]*advantage[idx]
            actor_loss.append(loss_idx*-1)
        actor_loss = torch.stack(actor_loss).mean()
        #print('ACTOR LOSS: ', actor_loss)
        return actor_loss

    def generate_video(self, max_frame=1000):
        # Generating the video multiple times with random initial states instead of just oneand saving them in folder structure according to the trails
        for i in range(20):
            self.env = gym.make(
                self.params['env_name'], render_mode='rgb_array_list')
            obs, _ = self.env.reset()
            for _ in range(max_frame):
                action_idx, log_prob = self.actor_net(torch.tensor(
                    obs, dtype=torch.float32, device=get_device()))
                obs, reward, terminated, truncated, info = self.env.step(
                    self.agent.action_space[action_idx.item()])
                if terminated or truncated:
                    break
            save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3]+'/'+self.params['exp_name'][-2:], name_prefix=(self.params['exp_name'])+'_video'+str(i),fps=self.env.metadata['render_fps'], step_starting_index=1, episode_index=1)



# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        # TODO: Define the actor net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax()
        )

    def forward(self, obs):
        # TODO: Forward pass of actor net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        probabilities = self.ff_net(obs)
        distribution = Categorical(probabilities)
        action_index = distribution.sample()
        log_prob = distribution.log_prob(action_index)
        return action_index, log_prob


# CLass for actor-net
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # TODO: Define the critic net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        # TODO: Forward pass of critic net
        # HINT: (get state value from the network using the current observation)
        state_value = self.ff_net(obs)
        return state_value


# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [
            action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {
                'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32,
                                   device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(
                    self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {
            'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(
                torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(
                torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer


class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(
            input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(
            input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(
            params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(
                    action)
                if terminated or truncated:
                    self.epsilon = max(
                        self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    #next_obs = None
                    self.replay_memory.push(
                        obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(
                        f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(
                    obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video(5000)
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability.
        with torch.no_grad():
            qValues = self.q_net(torch.tensor(obs, device=get_device()))
            if(np.random.random_sample() < self.epsilon):
                return self.env.action_space.sample()
            else:
                maxQValue = qValues.max()
                actionWithMaxQValue = torch.where(
                    qValues == maxQValue)[0][0].item()

                return actionWithMaxQValue

    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        # TODO: Update Q-net
        # HINT: You should draw a batch of random samples from the replay buffer
        # and train your Q-net with that sampled batch.

        observations, actions, rewards, next_observations, statuses = self.replay_memory.sample(
            self.params['batch_size'])

        observations = torch.tensor(observations, device=get_device())
        actions = torch.tensor(actions, device=get_device())
        rewards = torch.tensor(rewards, device=get_device())
        #next_observations = [obs if obs is not None else np.nan for obs in next_observations]

        next_observations = torch.tensor(
            next_observations, device=get_device())
        statuses = torch.tensor(statuses, device=get_device())

        # For Predicted Value
        # This contains qValues for all possible actions for each observation in observations
        qValues = self.q_net.forward(observations)

        #predicted_state_values = torch.tensor(qValues[range(self.params['batch_size']), i] for i in actions)
        # print(qValues[1])
        # print(actions[1])
        predicted_state_values = qValues[range(
            self.params['batch_size']), actions]
        # print(predicted_state_values[1])
        # For Target Value

        with torch.no_grad():
            targetQValues = self.target_net(next_observations)

        target_values = targetQValues.max(1)[0]
        target_values[statuses == False] = 0
        # print(target_values)
        target_values = rewards + self.params['gamma'] * target_values

        # operation = lambda x: torch.tensor(round(x.item()))
        # new_tensor_list = [operation(x) for x in target_values]
        # self.updateSteps += 1
        # print('Update Step: ', self.updateSteps)
        # print('Predicted Value:', predicted_state_values)
        # print('Target Value:', new_tensor_list)

        criterion = nn.SmoothL1Loss()
        q_loss = criterion(predicted_state_values.unsqueeze(
            1), target_values.unsqueeze(1))
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (
                1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        # Generating the video multiple times with random initial states instead of just oneand saving them in folder structure according to the trails
        print('MAX FRAME:', max_frame)
        for i in range(20):
            self.env = gym.make(
                self.params['env_name'], render_mode='rgb_array_list')
            self.epsilon = 0.0
            obs, _ = self.env.reset()
            for _ in range(max_frame):
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(
                    action)
                if terminated or truncated:
                    break
            save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3]+'/'+self.params['exp_name'][-2:], name_prefix=(self.params['exp_name'])+'_video'+str(i),fps=self.env.metadata['render_fps'], step_starting_index=1, episode_index=1)



class ReplayMemory:
    # TODO: Implement replay buffer
    # HINT: You can use python data structure deque to construct a replay buffer
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append([args[0], args[1], args[2], args[3], args[4]])

    def sample(self, n_samples):
        samples = random.sample(self.buffer, n_samples)
        observations, actions, rewards, next_observations, statuses = list(
        ), list(), list(), list(), list()
        for sample in samples:
            observations.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_observations.append(sample[3])
            statuses.append(sample[4])
        return observations, actions, rewards, next_observations, statuses


class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network from HW1 but we are not using Softmax layer at the end as this network gives the Q value but not the probabilities unlike PG Method
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        return self.ff_net(obs)
